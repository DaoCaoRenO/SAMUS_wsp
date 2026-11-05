#!/usr/bin/env bash
# =============================================================================
# run_multiorgan_exp.sh — USIC 多器官跨域对比实验编排（A+B 串行，单卡）
#   A: SAMUS-main + USIC(无甲状腺) + TN3K → 零样本跨域 DDTIz/PTz
#   B: blsamus + USIC(纯无甲状腺，无TN3K) → 零样本跨域 DDTIz/PTz
#   评测: rag_segment --lambda 0 --tag multi_tn3k/bl_nothy 产 crossdomain_results 行
#   最后刷两张总表
# 用法：setsid nohup bash run_multiorgan_exp.sh > multiorgan_chain.log 2>&1 &
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
SAMUS_DIR="/home/linux/Downloads/KTD/SAMUS-main"
BL_DIR="/home/linux/Downloads/ForPaper/code2026-10-30/code/blsamus"
CDJSONL="$SAMUS_DIR/crossdomain_results.jsonl"
BANK="$SAMUS_DIR/rag_bank/tn3k_bank.pt"
BS=8
# SAMUS-main 的 ep270 仅做 bank 用，不用作 base ckpt

cd "$SAMUS_DIR" || exit 1
echo "[multiorgan] START $(date)"

# ---- [0] 确认数据已准备（缺则跑 prepare）----
if [ ! -d "dataset/SAMUS/USIC_BUSBRA/img" ]; then
  echo "[multiorgan] [0] data not prepared → running prepare_usic_data.py"
  python3 prepare_usic_data.py
fi

# ---- [1] 冒烟 A (1 epoch，验证不炸) ----
echo "[multiorgan] [1] Smoke A (1 epoch) $(date)"
python train.py --modelname SAMUS --task MultiOrgan --epochs 1 --out_tag smoke_A 2>&1 | tail -5
if [ $? -ne 0 ]; then echo "[multiorgan] SMOKE A FAILED"; exit 1; fi
echo "[multiorgan] Smoke A OK $(date)"

# ---- [2] 正式 A (100 epoch) ----
echo "[multiorgan] [2] TRAIN A START $(date)"
python train.py --modelname SAMUS --task MultiOrgan --epochs 100 --out_tag multiA 2>&1 | tee train_multiA.log | tail -3
echo "[multiorgan] [2] TRAIN A DONE rc=$? $(date)"
CKPT_A="checkpoints/MultiOrgan_multiA/SAMUS_best.pth"
if [ ! -f "$CKPT_A" ]; then
  echo "[multiorgan] ERROR: A ckpt $CKPT_A not found"; exit 1
fi

# ---- [3] 训练 B (blsamus 无甲状腺 100 epoch) ----
echo "[multiorgan] [3] TRAIN B START $(date)"
cd "$BL_DIR" || exit 1
bash baseline_nothy.sh 2>&1 | tee train_nothy.log | tail -5
echo "[multiorgan] [3] TRAIN B DONE rc=$? $(date)"
# 找 best checkpoint
BL_BEST=$(ls -t exp_out/samus_b_nothy/best_model_*.pth 2>/dev/null | head -1)
if [ -z "$BL_BEST" ]; then echo "[multiorgan] ERROR: B ckpt not found"; exit 1; fi
echo "[multiorgan] B best: $BL_BEST"

# ---- [4] 提取 B 权重到 SAMUS 格式 ----
cd "$SAMUS_DIR" || exit 1
EXTRACTED="checkpoints/bl_nothy_extracted.pth"
echo "[multiorgan] [4] Extract B weights"
python3 extract_blsamus_ckpt.py "$BL_DIR/$BL_BEST" "$EXTRACTED"

# ---- [5] 评测 A: rag_segment --lambda 0 --tag multi_tn3k ----
echo "[multiorgan] [5] EVAL A on DDTIz/PTz $(date)"
for ds in DDTIz PTz; do
  python3 rag_segment.py --task "$ds" --split test --ckpt "$CKPT_A" --bank "$BANK" \
    --lambda 0 --k 1 --batch $BS --out rag_results.jsonl --emit-table "$CDJSONL" --tag multi_tn3k
done

# ---- [6] 评测 B: rag_segment --lambda 0 --tag bl_nothy ----
echo "[multiorgan] [6] EVAL B on DDTIz/PTz $(date)"
for ds in DDTIz PTz; do
  python3 rag_segment.py --task "$ds" --split test --ckpt "$EXTRACTED" --bank "$BANK" \
    --lambda 0 --k 1 --batch $BS --out rag_results.jsonl --emit-table "$CDJSONL" --tag bl_nothy
done

# ---- [7] 也评 A/B 在 TN3K 域内（了解多器官训练对源域的影响）----
echo "[multiorgan] [7] EVAL A/B on TN3K (domain-in) $(date)"
python3 rag_segment.py --task TN3K --split test --ckpt "$CKPT_A" --bank "$BANK" \
  --lambda 0 --k 1 --batch $BS --out rag_results.jsonl --emit-table "$CDJSONL" --tag multi_tn3k
python3 rag_segment.py --task TN3K --split test --ckpt "$EXTRACTED" --bank "$BANK" \
  --lambda 0 --k 1 --batch $BS --out rag_results.jsonl --emit-table "$CDJSONL" --tag bl_nothy

# ---- [8] 刷新两张总表 ----
echo "[multiorgan] [8] Refresh tables $(date)"
python3 make_crossdomain_table.py
python3 make_bl_comparison.py

echo "[multiorgan] ALL DONE $(date)"
