#!/usr/bin/env bash
# =============================================================================
# run_shift50_pair.sh — Phase3B §52 对照: shift50 vs base50 同图 paired (rag_segment λ=0)
#   补齐 run_train_shift.sh 的绝对值评测(test.py)之外的**配对**判据：同一确定性
#   loader 下, 分别用 rag_segment --lambda 0 跑 shift50 与 base50 逐图 Dice, 按
#   image_name 匹配做配对 Wilcoxon(§11 判据: ΔDice>0 且 p<0.05、SEN/SPC 无显著恶化)。
#   与运行中的 run_train_shift.sh 解耦：本脚本先等其 "[shift] ALL DONE" 且 GPU 空闲,
#   再评测, 避免显存竞争; 不改写总表(shift50 绝对行由 run_train_shift.sh 落)。
#   用法：setsid nohup bash run_shift50_pair.sh > shift50_pair_chain.log 2>&1 &
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

BS=8
BASE50="checkpoints/TN3K_base50/SAMUS_best.pth"
SHIFT50="checkpoints/TN3K_shift50/SAMUS_best.pth"
CHAIN="train_shift_chain.log"
STUDY="shift50_pair_results.jsonl"

echo "[pair50] START $(date)  waiting for shift50 training chain to finish"

# ---- [0] 等 run_train_shift.sh 打印 ALL DONE，且 GPU 上前台任务退出 ----
while ! grep -q "\[shift\] ALL DONE" "$CHAIN" 2>/dev/null; do
  sleep 60
done
while pgrep -f "out_tag shift50|test.py --modelname" >/dev/null 2>&1; do sleep 20; done
echo "[pair50] shift50 链已完成且 GPU 空闲 $(date)"

if [ ! -f "$SHIFT50" ]; then echo "[pair50] 缺 $SHIFT50，终止"; exit 1; fi
[ -f "$BASE50" ] || { echo "[pair50] WARN 缺 $BASE50，无法配对"; exit 1; }

rm -f "$STUDY"

run() {  # run <ds> <tag> <ckpt>
  local ds="$1" tag="$2" ckpt="$3"
  echo "==== rag_segment λ=0  ds=$ds tag=$tag ckpt=$(basename "$(dirname "$ckpt")") ===="
  python3 rag_segment.py --task "$ds" --split test --ckpt "$ckpt" \
    --lambda 0 --batch "$BS" --out "$STUDY" --tag "$tag"
}

for ds in DDTIz PTz TN3K; do
  run "$ds" base50  "$BASE50"
  run "$ds" shift50 "$SHIFT50"
done

# ---- [1] 同图配对分析 (§52) ----
echo "[pair50] same-image paired base50 -> shift50 $(date)"
python3 freq_adapt_paired.py "$STUDY" --pair base50 shift50

echo "[pair50] ALL DONE $(date)"
