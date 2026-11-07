#!/usr/bin/env bash
# =============================================================================
# run_hf_align_fine.sh — 口径对齐的「每域精调 γ」变体（run_hf_align.sh 的姊妹链）
#   run_hf_align.sh 用**统一** gain=0.60 建三集（tag hfal_e6/hfal_head6）；本链改用
#   磁盘 9 点扫描里**每域各自的实测峰**：DDTI 0.55(63.49) / PT 0.65(63.93)，
#   这才是 63.6 基线原本的那一档，因此本链的 head×E 才是与 63.6 严格同尺的头。
#   tag 用无后缀的 hfal_e / hfal_head（与 hfal_e6/hfal_head6 在总表里并列共存）。
#   每域独立后缀目录 {ds}_hf_g055 / {ds}_hf_g065 → 不覆写历史 _hf、也不与 g06 互撞。
#   判据同 §11：同图 per-image Wilcoxon，ΔDice>0 且 p<0.05，SEN 不显著恶化。
#   用法：setsid nohup bash run_hf_align_fine.sh > hf_align_fine.log 2>&1 &
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

CKPT="checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth"
BANK="rag_bank/tn3k_bank.pt"
HEAD_CKPT="checkpoints/TN3K_rag50/SAMUS_best.pth"
HIDDEN=32; K=5; RETR=roi; BS=8
CDJSONL="crossdomain_results.jsonl"
STUDY="hf_align_fine_results.jsonl"

echo "[fine] START $(date)  每域最优 gain: DDTIz=0.55 PTz=0.65"
rm -f "$STUDY"

# ds|gain|suffix —— suffix 必须含 _hf 前缀(hf_boost --suffix 只收其后半段)
for spec in "DDTIz|0.55|_hf_g055" "PTz|0.65|_hf_g065"; do
  IFS='|' read -r ds g sfx <<< "$spec"
  if [ "${SKIP_BUILD:-0}" != "1" ]; then
    echo "[fine] build $ds gain=$g -> ${ds}${sfx}"
    python hf_boost.py --gain "$g" --ratio 0.25 --tasks "$ds" --suffix "${sfx#_hf}" || exit 1
  fi
  [ -d "dataset/SAMUS/${ds}${sfx}/img" ] || { echo "[fine] 缺 dataset/SAMUS/${ds}${sfx}，终止"; exit 1; }

  echo "==== ds=$ds tag=hfal_e  (磁盘E ${sfx}, 无头) ===="
  python3 rag_segment.py --task "${ds}${sfx}" --split test --ckpt "$CKPT" --bank "$BANK" \
    --lambda 0 --batch "$BS" --ds-key "$ds" --out "$STUDY" --tag hfal_e \
    --emit-table "$CDJSONL"

  echo "==== ds=$ds tag=hfal_head  (磁盘E ${sfx} × PCR头) ===="
  python3 rag_segment.py --task "${ds}${sfx}" --split test --ckpt "$CKPT" --bank "$BANK" \
    --lambda 0 --batch "$BS" --ds-key "$ds" --out "$STUDY" --tag hfal_head \
    --emit-table "$CDJSONL" --head_ckpt "$HEAD_CKPT" --head_hidden "$HIDDEN" --k "$K" --retr "$RETR"
done

# ---- 同图配对：head 在精调 E 之上的边际 ----
python3 freq_adapt_paired.py "$STUDY" --pair hfal_e hfal_head || echo "[fine] WARN paired"

# ---- 精调口径的 head×E 跨域均值 vs 63.6 ----
python3 - "$STUDY" <<'PY' || echo "[fine] WARN verdict"
import sys, json, numpy as np
XDOM = ["DDTIz", "PTz"]
m, gain = {}, {}
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    per = r.get("per", {})
    if r.get("tag") and per.get("names"):
        m[(r["ds"], r["tag"])] = np.array(per["rag_dice"], dtype=float)
        gain[(r["ds"], r["tag"])] = r.get("ds_raw", "")
print("\n===== [fine] 每域精调 gain 口径 (DDTI 0.55 / PT 0.65) =====")
tot = {}
for t in ["hfal_e", "hfal_head"]:
    row = [m[("DDTIz", t)].mean(), m[("PTz", t)].mean()]
    tot[t] = float(np.mean(row))
    print("  %-10s DDTI=%.2f PT=%.2f 跨域=%.2f" % (t, row[0], row[1], tot[t]))
print("  参照 63.6 ；head 边际 = %+.2f ；head×E %s 63.6"
      % (tot["hfal_head"] - tot["hfal_e"],
         "≥(过 bar)" if tot["hfal_head"] > 63.6 else "<(未过)"))
PY

if [ "${SKIP_TABLE:-0}" != "1" ]; then
  echo "[fine] refresh tables $(date)"
  python3 make_crossdomain_table.py
  python3 make_bl_comparison.py
fi
echo "[fine] ALL DONE $(date)"
