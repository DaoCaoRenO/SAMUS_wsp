#!/usr/bin/env bash
# =============================================================================
# run_hf_align.sh — 方案E「磁盘口径」对齐：让 head×E 与 63.6 站在同一测量尺上
#   背景(§风险/踩坑第2条)：Phase1 的 head×E 跨域均值 62.3 是**内存版** hfa_boost
#   (loader 输出张量上乘 gain)，而 63.6 基线来自**磁盘版** hf_boost.py
#   (uint8 + JPEG 重编码后落盘再读)。两者对同一 gain 的净 Dice 不同
#   (DDTI 内存 +2.96 vs 磁盘 +5.16)，所以"62.3 < 63.6 未过 bar"里混了口径差。
#   做法：用 hf_boost.py --suffix 另建独立目录 {ds}_hf_g06(gain=0.60, ratio=0.25,
#   即磁盘扫描里 DDTI 63.36/PT 63.78 的那一档)，然后**同一 rag_segment 链路、
#   λ=0、同图**跑三路：
#     hfal_base   = 原图 ep270             (对照, 应与 test.py 基线 58.32/62.15 一致)
#     hfal_e6     = 统一 gain 磁盘集 + 无头     (应与磁盘 63.36/63.78 一致 → 评测器无偏移)
#     hfal_head6  = 统一 gain 磁盘集 + 可学习PCR头 (head 在 E 之上的**边际**增量)
#   tag 后缀 '6' = 三集共用同一个 gain(=$GAIN)；每域各自最优 gain 的变体由
#   run_hf_align_fine.sh 产出，用无后缀的 hfal_e/hfal_head 两个 tag。
#   再按 image_name 跨行配对(§11 判据: ΔDice>0 且 Wilcoxon p<0.05、SEN 不显著恶化)。
#   不覆写历史 _hf 数据(那是扫描最后一次 gain=1.2 的残留)。
#   用法：setsid nohup bash run_hf_align.sh > hf_align.log 2>&1 &
#         (纯推理无训练；SKIP_TABLE=1 只跑分析不刷表；SKIP_BUILD=1 复用已建数据)
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

CKPT="checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth"
BANK="rag_bank/tn3k_bank.pt"
HEAD_CKPT="checkpoints/TN3K_rag50/SAMUS_best.pth"
HIDDEN=32; K=5; RETR=roi; BS=8
GAIN="${GAIN:-0.6}"                       # 磁盘扫描的最强档(γ*≈0.61 取整)
SUFFIX="_hf_g${GAIN/./}"                  # 0.6 -> _hf_g06 (hf_boost --suffix 只收 '_g06' 部分)
CDJSONL="crossdomain_results.jsonl"
STUDY="hf_align_results.jsonl"

echo "[hfal] START $(date)  gain=$GAIN suffix=$SUFFIX head=$HEAD_CKPT"

# ---- [0] 建磁盘增强集(独立后缀, 不动历史 _hf) ----
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo "[hfal] build hf_boost gain=$GAIN ratio=0.25 -> *${SUFFIX}"
  python hf_boost.py --gain "$GAIN" --ratio 0.25 --suffix "${SUFFIX#_hf}" || exit 1
else
  echo "[hfal] SKIP_BUILD=1，复用已有 *${SUFFIX} 数据"
fi
for d in TN3K DDTIz PTz; do
  [ -d "dataset/SAMUS/${d}${SUFFIX}/img" ] || { echo "[hfal] 缺 dataset/SAMUS/${d}${SUFFIX}，终止"; exit 1; }
done

# ---- [1] 三路矩阵(同链路/λ=0/同图; --ds-key 把增强集归回基础 ds 以便跨行配对) ----
rm -f "$STUDY"

run() {  # run <ds> <tag> <emit(1/0)> [rag_segment args...]
  local ds="$1" tag="$2" emit="$3"; shift 3
  local extra=()
  [ "$emit" = "1" ] && extra=(--emit-table "$CDJSONL")
  echo "==== ds=$ds tag=$tag  $* ===="
  python3 rag_segment.py --split test --ckpt "$CKPT" --bank "$BANK" --batch "$BS" \
    --lambda 0 --ds-key "$ds" --out "$STUDY" --tag "$tag" "${extra[@]}" "$@"
}

for ds in TN3K DDTIz PTz; do
  run "$ds" hfal_base   0 --task "$ds"                                     # 对照
  run "$ds" hfal_e6     1 --task "${ds}${SUFFIX}"                          # 磁盘E, 无头
  run "$ds" hfal_head6  1 --task "${ds}${SUFFIX}" \
      --head_ckpt "$HEAD_CKPT" --head_hidden "$HIDDEN" --k "$K" --retr "$RETR"    # 磁盘E×头
done

# ---- [2] 跨行 per-image 配对 (§11 判据) ----
echo "[hfal] paired analysis $(date)"
python3 freq_adapt_paired.py "$STUDY" --pair hfal_base hfal_e6     || echo "[hfal] WARN pair1"
python3 freq_adapt_paired.py "$STUDY" --pair hfal_e6   hfal_head6   || echo "[hfal] WARN pair2"
python3 freq_adapt_paired.py "$STUDY" --pair hfal_base hfal_head6  || echo "[hfal] WARN pair3"

# ---- [3] 对齐判定：磁盘口径 head×E 的跨域均值 vs 63.6 ----
python3 - "$STUDY" "$GAIN" <<'PY' || echo "[hfal] WARN verdict"
import sys, json, numpy as np
XDOM = ["DDTIz", "PTz"]
m = {}
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    per = r.get("per", {})
    if r.get("tag") and per.get("names"):
        m[(r["ds"], r["tag"])] = np.array(per["rag_dice"], dtype=float)

def mean(ds, tag):
    v = m.get((ds, tag))
    return float(v.mean()) if v is not None else float("nan")

def xd(tag):
    vals = [mean(ds, tag) for ds in XDOM]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")

print("\n===== [hfal] 磁盘口径对齐判定 (统一 gain=%s; 跨域=DDTI/PT 均值) =====" % sys.argv[2])
for t in ["hfal_base", "hfal_e6", "hfal_head6"]:
    print("  %-11s DDTI=%.2f PT=%.2f 跨域=%.2f" % (t, mean("DDTIz", t), mean("PTz", t), xd(t)))
print("  参照：磁盘方案E 最好成绩 63.6 ；内存口径 head×E 62.3")
print("  [口径一致性] hfal_e6 跨域=%.2f 应≈63.6(差<0.5 说明 rag_segment 与 test.py 同尺)" % xd("hfal_e6"))
print("  [head 边际]  hfal_head6 - hfal_e6 = %+.2f" % (xd("hfal_head6") - xd("hfal_e6")))
ver = "过 bar(磁盘口径 head×E ≥ 63.6 → 值得重训/写进主线)" if xd("hfal_head6") > 63.6 \
      else "仍未过 bar(磁盘口径 head×E < 63.6 → §38 跳过 2B 的结论对齐后依然成立)"
print("  [§38 对齐判定] head×E=%.2f vs 63.6 -> %s" % (xd("hfal_head6"), ver))
PY

# ---- [4] 刷两张总表 ----
if [ "${SKIP_TABLE:-0}" != "1" ]; then
  echo "[hfal] refresh tables $(date)"
  python3 make_crossdomain_table.py
  python3 make_bl_comparison.py
fi

echo "[hfal] ALL DONE $(date)"
