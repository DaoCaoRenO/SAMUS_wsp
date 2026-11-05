#!/usr/bin/env bash
# =============================================================================
# run_rag_head.sh — RAG 可训练原型条件化精炼头(PCR head) 全流程编排（对应计划 §6）
#   假设（承接免训练四轮定案）：可学习的检索条件化头，能学会「base 不可靠时相信原型
#   先验到什么程度、在哪里补」，把 SEN 召回增益转成净 Dice。
#   范式：主干(ep270)全冻结、只训 PCR 头(末层零初始化恒等)、域偏移模拟迫使头依赖检索。
#   链路：建库(缺则补) → 域偏移量级自检 → 冒烟1轮(校验 pcr 键+val dice) → 正式50轮
#         → 零样本跨域评测(rag_segment --head_ckpt, tag=raghead, 含配对) → 刷两张总表。
#   对照：base(ep270_300) 与 固定λ先验(rag) 已在总表；本轮判「学习能否把召回转成 Dice」。
# 用法：setsid nohup bash run_rag_head.sh > rag_head_chain.log 2>&1 &
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

BS=8
HIDDEN=32
K=5
RETR=roi
SHIFT=1
SHIFT_STR=1.0
CKPT="checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth"
BANK="rag_bank/tn3k_bank.pt"
CDJSONL="crossdomain_results.jsonl"
RAGARGS="--rag_head 1 --rag_head_hidden $HIDDEN --rag_k $K --rag_retr $RETR --rag_shift $SHIFT \
         --rag_shift_strength $SHIFT_STR --rag_bank $BANK --rag_freeze_all 1 --rag_base_ckpt $CKPT"

echo "[raghead] START $(date)  (hidden=$HIDDEN k=$K retr=$RETR shift=$SHIFT str=$SHIFT_STR)"

# ---- [0] 记忆库（缺则建）----
if [ ! -f "$BANK" ]; then
  echo "[raghead] bank 缺失，构建 $BANK $(date)"
  python3 rag_build_bank.py --ckpt "$CKPT" --split train --out "$BANK"
fi

# ---- [1] 域偏移量级自检：强 shift 下 base dice 应显著低于源域(否则头会忽略先验) ----
echo "[raghead] domain-shift magnitude check $(date)"
python3 - <<'PY'
import torch, numpy as np
import rag_common as R
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, opt, ns = R.build_model(task="TN3K", load_path="checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth")
loader = R.make_loader(opt, "val", batch_size=8)
def dice_of(shift_s, limit=8):
    ds = []
    for bi, dp in enumerate(loader):
        if bi >= limit: break
        imgs = dp["image"].to(dev, dtype=torch.float32)
        if shift_s > 0:
            imgs = R.shift_image(imgs, strength=shift_s)
        gt = (dp["label"].to(dev, dtype=torch.float32)[:, 0] == 1).cpu().numpy()
        logits = R.model_base_logits(model, imgs)
        seg = (torch.sigmoid(logits)[:, 0] > 0.5).cpu().numpy()
        for j in range(seg.shape[0]):
            inter = (seg[j] & gt[j]).sum(); s = seg[j].sum() + gt[j].sum()
            ds.append(2*inter/max(s,1))
    return float(np.mean(ds))*100
clean = dice_of(0.0); hard = dice_of(1.5)
print(f"[shift-check] base val dice  clean={clean:.2f}  strong_shift={hard:.2f}  Δ={hard-clean:+.2f}")
if hard > clean - 3.0:
    print("[shift-check] WARN: 强 shift 未显著打低 base，头可能学会忽略先验(考虑加大 SHIFT_STR)")
else:
    print("[shift-check] OK: 强 shift 显著退化 base → 迫使头依赖检索先验")
PY

# ---- [2] 冒烟 1 轮：验证不炸、出 val dice、权重含 pcr 键 ----
python train.py --modelname SAMUS --task TN3K --epochs 1 --out_tag rag_smoke $RAGARGS > train_rag_smoke.log 2>&1
if ! grep -q "val dice" train_rag_smoke.log; then
  echo "[raghead] 冒烟失败(无 val dice)，终止。尾部："; tail -30 train_rag_smoke.log; exit 1
fi
SMOKE_CKPT="checkpoints/TN3K_rag_smoke/SAMUS_best.pth"
NPCR=$(python3 -c "import torch;sd=torch.load('$SMOKE_CKPT',map_location='cpu');print(sum(1 for k in sd if k.startswith('pcr.')))" 2>/dev/null || echo 0)
if [ "$NPCR" -lt 1 ]; then
  echo "[raghead] 冒烟权重无 pcr.* 键(NPCR=$NPCR)，终止。"; tail -20 train_rag_smoke.log; exit 1
fi
echo "[raghead] 冒烟通过 $(date)  pcr键数=$NPCR"; grep -E "RAG PCR|RAG base|val dice|Total_params" train_rag_smoke.log | tail -6

# ---- [3] 正式 50 轮 ----
echo "[raghead] 正式训练 START $(date)"
python train.py --modelname SAMUS --task TN3K --epochs 50 --out_tag rag50 $RAGARGS > train_rag50.log 2>&1
echo "[raghead] 训练完成 rc=$? $(date)"

# ---- [4] 零样本跨域评测（统一走 rag_segment --head_ckpt，含配对；upsert tag=raghead）----
HEAD_CKPT="checkpoints/TN3K_rag50/SAMUS_best.pth"
if [ -f "$HEAD_CKPT" ]; then
  for ds in TN3K DDTIz PTz; do
    echo "[raghead] eval head on $ds $(date)"
    python3 rag_segment.py --task "$ds" --split test --ckpt "$CKPT" --bank "$BANK" \
      --head_ckpt "$HEAD_CKPT" --head_hidden "$HIDDEN" --k "$K" --retr "$RETR" \
      --batch "$BS" --out rag_results.jsonl --emit-table "$CDJSONL"
  done
else
  echo "[raghead] 缺权重 $HEAD_CKPT，跳过评测"
fi

# ---- [5] 刷新两张总表 ----
echo "[raghead] refresh tables $(date)"
python3 make_crossdomain_table.py
python3 make_bl_comparison.py
echo "[raghead] ALL DONE $(date)"
