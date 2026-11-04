#!/usr/bin/env bash
# =============================================================================
# coarse-to-fine（PBR 渐进式边界精炼头）：对照组 = base50，同 seed/同初始化/50轮
#   冒烟1轮 -> 校验权重含 pbr_head -> 正式50轮 -> 自动跨域测试(TN3K/DDTIz/PTz) -> 追加 jsonl -> 更新表
#   c2f50 = base + PBR 头(零初始化残差,恒等接入) + 深监督(粗纯Dice / 细Dice+边缘)。
#   与 base50 干净 A/B：RNG 隔离已验证，dataloader 顺序逐位相同，唯一变量是 PBR 头。
# 用法：setsid nohup bash run_c2f.sh > c2f_chain.log 2>&1 &
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

BS=8
ITERS=1        # 精炼迭代次数（1=单级 coarse->fine；>1 即多级渐进推理，留作后续扫描）
HIDDEN=32      # PBR 头隐藏通道
EDGE_W=0.5     # 细阶段边缘损失权重（沿用 edge50 标定值 w=0.5）
EDGE_RAMP=10   # 边缘项前10轮线性升温
CDJSONL="crossdomain_results.jsonl"
C2F="--c2f 1 --c2f_iters $ITERS --c2f_hidden $HIDDEN --c2f_edge_w $EDGE_W --c2f_edge_ramp $EDGE_RAMP"

echo "[c2f] START $(date)  (iters=$ITERS hidden=$HIDDEN edge_w=$EDGE_W ramp=$EDGE_RAMP)"

# ---- 冒烟 1 轮：验证不炸、能出 val dice、权重含 pbr_head ----
python train.py --modelname SAMUS --task TN3K --epochs 1 --out_tag c2f_smoke $C2F > train_c2f_smoke.log 2>&1
if ! grep -q "val dice" train_c2f_smoke.log; then
  echo "[c2f] 冒烟失败，终止。尾部："; tail -25 train_c2f_smoke.log; exit 1
fi
SMOKE_CKPT="checkpoints/TN3K_c2f_smoke/SAMUS_best.pth"
NPBR=$(python3 -c "import torch;sd=torch.load('$SMOKE_CKPT',map_location='cpu');print(sum(1 for k in sd if 'pbr_head' in k))" 2>/dev/null || echo 0)
if [ "$NPBR" -lt 1 ]; then
  echo "[c2f] 冒烟权重无 pbr_head 键(NPBR=$NPBR)，终止。"; tail -15 train_c2f_smoke.log; exit 1
fi
echo "[c2f] 冒烟通过 $(date)  pbr_head键数=$NPBR"; grep -E "CoarseToFine|PBR|val dice|Total_params" train_c2f_smoke.log | tail -6

# ---- 正式 50 轮 ----
echo "[c2f] 正式训练 START $(date)"
python train.py --modelname SAMUS --task TN3K --epochs 50 --out_tag c2f50 $C2F > train_c2f50.log 2>&1
echo "[c2f] 训练完成 rc=$? $(date)"

# ---- 自动跨域测试 ----
CKPT="checkpoints/TN3K_c2f50/SAMUS_best.pth"
if [ -f "$CKPT" ]; then
  for ds in TN3K DDTIz PTz; do
    echo "[c2f] test c2f50 on $ds $(date)"
    out=$(python test.py --modelname SAMUS --task "$ds" --load_path "$CKPT" --batch_size "$BS" \
          --c2f 1 --c2f_iters "$ITERS" --c2f_hidden "$HIDDEN" 2>/dev/null \
          | tr '\r' '\n' | grep '^RESULT_JSON ' | head -1 | sed 's/^RESULT_JSON //')
    if [ -n "$out" ]; then
      echo "$out" | python3 -c "import sys,json;d=json.load(sys.stdin);d['tag']='c2f50';d['ds']='$ds';print(json.dumps(d))" >> "$CDJSONL"
      echo "   -> dice=$(echo "$out" | python3 -c 'import sys,json;print("%.2f"%json.load(sys.stdin)["dice"])')"
    else
      echo "   [FAIL] c2f50 $ds"
    fi
  done
  python3 make_crossdomain_table.py
else
  echo "[c2f] 缺权重 $CKPT，跳过测试"
fi
echo "[c2f] ALL DONE $(date)"
