#!/usr/bin/env bash
# =============================================================================
# run_freq_adapt.sh — 方案E 逐图自适应γ 校准 + head×E 可加性 (Phase 1 免训练评测)
#   目标(对应计划 §1B/§1C)：在免训练/只读现有 head 的前提下，把「自适应γ、固定γ、
#   head、head×固定γ、head×自适应γ」放到同一 rag_segment 链路(同 loader/同图)跑三集，
#   再用跨行 per-image 配对(按 image_name 匹配)判定：
#     §37a 自适应γ paired 显著正且 ≥ 固定γ → 确立自校准方案E 为主线；
#     §38  head×E ≈ head + E 近可加(跨域均值 >63.6 且显著) → 进 Phase2 重训。
#   链路：算/缓存 E_src_ref → 3 集 × 6 配置矩阵 → 跨行 paired → 刷两张总表 → tail。
#   用法：setsid nohup bash run_freq_adapt.sh > run_freq_adapt.log 2>&1 &
#         (GPU 分钟级、无训练；SKIP_TABLE=1 可只跑分析不刷表)
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

CKPT="checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth"
BANK="rag_bank/tn3k_bank.pt"
HEAD_CKPT="checkpoints/TN3K_rag50/SAMUS_best.pth"
HIDDEN=32
K=5
RETR=roi
BS=8
RATIO=0.25                      # 自适应γ 高频半径(与固定γ/hf_boost 口径一致)
CDJSONL="crossdomain_results.jsonl"
STUDY="freq_adapt_results.jsonl"      # 本轮研究日志(含 per-image, 供跨行配对)
HFREF="rag_bank/hf_ref.json"          # E_src_ref 缓存

# 每域固定 γ*(=hf_boost.py measure 的 源域/目标域 高频能量比)
gamma_of() { case "$1" in
  DDTIz) echo 0.61 ;;
  PTz)   echo 0.65 ;;
  *)     echo 1.0 ;;
esac; }

echo "[freq] START $(date)  (ratio=$RATIO head=$HEAD_CKPT)"

# ---- [0] 记忆库(缺则建) ----
if [ ! -f "$BANK" ]; then
  echo "[freq] bank 缺失，构建 $BANK"
  python3 rag_build_bank.py --ckpt "$CKPT" --split train --out "$BANK"
fi

# ---- [1] 计算/缓存 E_src_ref (源域 TN3K train 高频能量中位数, 秒级~分钟级) ----
if [ ! -f "$HFREF" ]; then
  echo "[freq] compute E_src_ref -> $HFREF"
  python3 - "$HFREF" <<'PY'
import sys, json, rag_common as R
out = sys.argv[1]
ref = R.source_hf_ref(task="TN3K", split="train", ratio=0.25)
json.dump(ref, open(out, "w"))
print("[freq] E_src_ref:", ref)
PY
else
  echo "[freq] using cached $HFREF"
fi

# ---- [2] 评测矩阵 (每行同 loader/同图; --out 收 per-image, --emit-table 落总表) ----
rm -f "$STUDY"

run() {  # run <ds> [rag_segment args...]
  local ds="$1"; shift
  echo "==== ds=$ds  $* ===="
  python3 rag_segment.py --task "$ds" --split test --ckpt "$CKPT" --bank "$BANK" \
    --batch "$BS" --out "$STUDY" "$@"
}

for ds in TN3K DDTIz PTz; do
  GAM=$(gamma_of "$ds")
  HA=(--head_ckpt "$HEAD_CKPT" --head_hidden "$HIDDEN" --k "$K" --retr "$RETR")
  # 1) base：γ=1 纯对照(λ=0)
  run "$ds" --lambda 0 --tag "base"
  # 2) 固定γ(免训练, 无 head)
  run "$ds" --lambda 0 --stack-e "$GAM" --tag "efix"
  # 3) 自适应γ(免训练, 无 head) -> 表 tag e_adapt
  run "$ds" --lambda 0 --e-adapt "$RATIO" --e-ref "$HFREF" --emit-table "$CDJSONL" --tag e_adapt
  # 4) head(无 E) —— 可加性分解用 (同 raghead)
  run "$ds" "${HA[@]}" --lambda 0 --tag "head"
  # 5) head×固定γ -> 表 tag raghead_e (E=已知 +3 那档固定γ)
  run "$ds" "${HA[@]}" --lambda 0 --stack-e "$GAM" --emit-table "$CDJSONL" --tag raghead_e
  # 6) head×自适应γ(仅研究日志, 判是否优于 head×固定γ)
  run "$ds" "${HA[@]}" --lambda 0 --e-adapt "$RATIO" --e-ref "$HFREF" --tag "head_eadapt"
done

# ---- [3] 跨行 per-image 配对分析 (按 image_name 匹配同图, Wilcoxon) ----
echo "[freq] cross-row paired analysis $(date)"
python3 freq_adapt_paired.py "$STUDY" || echo "[freq] WARN paired 分析失败"

# ---- [4] 刷两张总表 ----
if [ "${SKIP_TABLE:-0}" != "1" ]; then
  echo "[freq] refresh tables $(date)"
  python3 make_crossdomain_table.py
  python3 make_bl_comparison.py
fi

echo "[freq] ALL DONE $(date)"
