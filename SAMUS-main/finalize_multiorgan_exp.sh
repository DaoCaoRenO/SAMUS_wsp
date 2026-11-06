#!/usr/bin/env bash
# =============================================================================
# finalize_multiorgan_exp.sh — B 训练完成后的一键收尾（评测 A/B + 刷两表）
#   前提: run_multiorgan_exp.sh [2] 训 A 已完成, B 训练 baseline_nothy.sh 已完成
#   本脚本负责原 run_multiorgan_exp.sh 里 [4]-[8] 手工重跑（因 B 曾崩溃导致链终止）
#
# 用法: setsid nohup bash finalize_multiorgan_exp.sh > finalize.log 2>&1 &
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
SAMUS_DIR="/home/linux/Downloads/KTD/SAMUS-main"
BL_DIR="/home/linux/Downloads/ForPaper/code2026-10-30/code/blsamus"
CDJSONL="$SAMUS_DIR/crossdomain_results.jsonl"
BANK="$SAMUS_DIR/rag_bank/tn3k_bank.pt"
CKPT_A="$SAMUS_DIR/checkpoints/MultiOrgan_multiA/SAMUS_best.pth"
BS=8

echo "[finalize] START $(date)"

# ---- [1] 校验 A/B ckpt 存在 ----
if [ ! -f "$CKPT_A" ]; then echo "[finalize] ERROR: A ckpt missing"; exit 1; fi
BL_BEST=$(ls -t "$BL_DIR/exp_out/samus_b_nothy/best_model_"*.pth 2>/dev/null | head -1)
if [ -z "$BL_BEST" ]; then echo "[finalize] ERROR: B ckpt missing"; exit 1; fi
echo "[finalize] A ckpt: $CKPT_A"
echo "[finalize] B ckpt: $BL_BEST"

# ---- [2] 评 A: rag_segment 在 SAMUS-main 256 输入下自然兼容 ----
cd "$SAMUS_DIR" || exit 1
echo "[finalize] [2] EVAL A on DDTIz/PTz/TN3K $(date)"
for ds in DDTIz PTz TN3K; do
  python3 rag_segment.py --task "$ds" --split test --ckpt "$CKPT_A" --bank "$BANK" \
    --lambda 0 --k 1 --batch $BS --out rag_results.jsonl \
    --emit-table "$CDJSONL" --tag multi_tn3k 2>&1 | tail -6
done

# ---- [3] 评 B: blsamus 224 输入, 用 omni_test.py (不 extract 因 shape 不兼容) ----
cd "$BL_DIR" || exit 1
echo "[finalize] [3] EVAL B via omni_test.py (DDTI/private_Thyroid/TN3K) $(date)"
# 注意: B 输出写独立 result_b_nothy.csv, 不覆盖官方 blsamus 基线 exp_out/result.csv
#       (make_bl_comparison.py 从 exp_out/result.csv 读 blsamus 域内基线行)
rm -f exp_out/result_b_nothy.csv
torchrun --nproc_per_node=1 --master_port=22347 omni_test.py \
  --root_path=./data \
  --cfg=./configs/samus_config.yaml \
  --output_dir=exp_out/samus_b_nothy \
  --batch_size=4 \
  --base_lr=1e-5 \
  --max_epochs=100 \
  --prompt \
  --result_csv=exp_out/result_b_nothy.csv 2>&1 | tee eval_b_crossdomain.log | tail -20

# ---- [4] 从 result_b_nothy.csv 提取 B 的 DDTI/private_Thyroid/TN3K 三个 dice → 写 jsonl ----
cd "$SAMUS_DIR" || exit 1
echo "[finalize] [4] Emit B rows to $CDJSONL $(date)"
python3 bl_nothy_to_jsonl.py \
  --result-csv "$BL_DIR/exp_out/result_b_nothy.csv" \
  --emit-table "$CDJSONL"

# ---- [5] 刷两张总表 ----
echo "[finalize] [5] Refresh tables $(date)"
python3 make_crossdomain_table.py
python3 make_bl_comparison.py

echo "[finalize] ALL DONE $(date)"
