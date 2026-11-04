#!/usr/bin/env bash
# =============================================================================
# 边缘增强 第1层：Sobel 空域边界损失（对照组 = base50，同 seed/同初始化/50轮）
#   冒烟1轮 -> 正式50轮 -> 自动跨域测试(TN3K/DDTIz/PTz) -> 追加 jsonl -> 更新汇总表
#   edge50 无 Freq_Adapter/无增广/无FFL，纯 base + 边缘损失，与 base50 干净 A/B。
# 用法：setsid nohup bash run_edge.sh > edge_chain.log 2>&1 &
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

BS=8
W=0.5       # 边缘损失权重（标定：edge/BCE≈1.2-1.7，w=0.5 使中后期约占主损失30-40%）
RAMP=10     # 前10轮线性升温，避免早期压倒主损失
CDJSONL="crossdomain_results.jsonl"

echo "[edge] START $(date)  (w=$W ramp=$RAMP)"

# ---- 冒烟 1 轮：验证不炸、能出 val dice ----
python train.py --modelname SAMUS --task TN3K --epochs 1 --out_tag edge_smoke \
       --edge_loss 1 --edge_loss_w "$W" --edge_loss_ramp "$RAMP" > train_edge_smoke.log 2>&1
if ! grep -q "val dice" train_edge_smoke.log; then
  echo "[edge] 冒烟失败，终止。尾部："; tail -25 train_edge_smoke.log; exit 1
fi
echo "[edge] 冒烟通过 $(date)"; grep -E "EdgeLoss|val dice|train loss" train_edge_smoke.log | tail -4

# ---- 正式 50 轮 ----
echo "[edge] 正式训练 START $(date)"
python train.py --modelname SAMUS --task TN3K --epochs 50 --out_tag edge50 \
       --edge_loss 1 --edge_loss_w "$W" --edge_loss_ramp "$RAMP" > train_edge50.log 2>&1
echo "[edge] 训练完成 rc=$? $(date)"

# ---- 自动跨域测试 ----
CKPT="checkpoints/TN3K_edge50/SAMUS_best.pth"
if [ -f "$CKPT" ]; then
  for ds in TN3K DDTIz PTz; do
    echo "[edge] test edge50 on $ds $(date)"
    out=$(python test.py --modelname SAMUS --task "$ds" --load_path "$CKPT" --batch_size "$BS" 2>/dev/null \
          | tr '\r' '\n' | grep '^RESULT_JSON ' | head -1 | sed 's/^RESULT_JSON //')
    if [ -n "$out" ]; then
      echo "$out" | python3 -c "import sys,json;d=json.load(sys.stdin);d['tag']='edge50';d['ds']='$ds';print(json.dumps(d))" >> "$CDJSONL"
      echo "   -> dice=$(echo "$out" | python3 -c 'import sys,json;print("%.2f"%json.load(sys.stdin)["dice"])')"
    else
      echo "   [FAIL] edge50 $ds"
    fi
  done
  python3 make_crossdomain_table.py
  python3 make_bl_comparison.py   # 跨口径性能对比总表(合并 blsamus+SAMUS)，自动写回 探索复盘记录.md
else
  echo "[edge] 缺权重 $CKPT，跳过测试"
fi
echo "[edge] ALL DONE $(date)"
