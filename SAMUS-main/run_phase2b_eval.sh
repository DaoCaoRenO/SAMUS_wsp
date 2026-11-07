#!/usr/bin/env bash
# =============================================================================
# run_phase2b_eval.sh — Phase2B 训练后的评测链（由 phase2b_watch.sh 在训练结束后触发）
#   为什么要独立出来：run_phase2b.sh 的冒烟取值正则写成 "val dice [0-9.]*"(空格)，
#   而 train.py 实际打印 "val dice:0.8348"(冒号) → SMOKE_DICE 解析成空串、护栏判定
#   抛 ValueError(仅 WARN、不致命，训练照常跑)。脚本运行中不可就地编辑(字节偏移)，
#   故评测段在此重写并修正：
#     · 主判据走**内存链路** --e-adapt(训练用的就是内存版 hfa_boost，唯一严格同尺的对照)
#     · 佐证走**磁盘链路**(E 集 × 新头) → 与 64.0~64.1(旧头×磁盘E) 直接可比
#     · 新头 tag = raghead_e_tr2(内存, 入总表) / 佐证行 tag = hfal_head2(磁盘, 入总表)
#   用法：setsid nohup bash run_phase2b_eval.sh > phase2b_eval.log 2>&1 &   (或经 watch)
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

CKPT="checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth"
BANK="rag_bank/tn3k_bank.pt"
OLD_HEAD="checkpoints/TN3K_rag50/SAMUS_best.pth"
NEW_HEAD="checkpoints/TN3K_rag50_e_tr/SAMUS_best.pth"
HIDDEN=32; K=5; RETR=roi; BS=8; RATIO=0.25
HFREF="rag_bank/hf_ref.json"
CDJSONL="crossdomain_results.jsonl"
MEM_STUDY="phase2b_results.jsonl"          # 内存链路 per-image
DISK_STUDY="phase2b_disk_results.jsonl"    # 磁盘口径 per-image
TAG2=raghead_e_tr2

echo "[p2be] START $(date)"
[ -f "$NEW_HEAD" ] || { echo "[p2be] 缺 $NEW_HEAD(训练未产出)，终止"; exit 1; }
NPCR=$(python3 -c "import torch;sd=torch.load('$NEW_HEAD',map_location='cpu');print(sum(1 for k in sd if k.startswith('pcr.')))")
BEST=$(grep -o "new best dice [0-9.]*" train_rag50_e_tr.log | tail -1)
echo "[p2be] new head pcr_keys=$NPCR  train $BEST  (base val dice≈0.835 为护栏)"
[ "$NPCR" -ge 1 ] || { echo "[p2be] 新权重无 pcr.* 键，终止"; exit 1; }

run_mem() {  # 内存链路: run_mem <ds> <tag> <emit> [extra...]
  local ds="$1" tag="$2" emit="$3"; shift 3
  local extra=(); [ "$emit" = "1" ] && extra=(--emit-table "$CDJSONL")
  echo "==== [mem] ds=$ds tag=$tag $* ===="
  python3 rag_segment.py --task "$ds" --split test --ckpt "$CKPT" --bank "$BANK" \
    --lambda 0 --batch "$BS" --out "$MEM_STUDY" --tag "$tag" "${extra[@]}" "$@"
}
run_disk() {  # 磁盘口径: 增强集 + --ds-key 归组, 不叠加内存 E
  local ds="$1" sfx="$2" tag="$3" extra4="${4:-}"
  echo "==== [disk] ds=$ds tag=$tag task=${ds}${sfx} ===="
  python3 rag_segment.py --task "${ds}${sfx}" --split test --ckpt "$CKPT" --bank "$BANK" \
    --lambda 0 --batch "$BS" --ds-key "$ds" --out "$DISK_STUDY" --tag "$tag" \
    --emit-table "$CDJSONL" --head_ckpt "$NEW_HEAD" --head_hidden "$HIDDEN" --k "$K" --retr "$RETR" $extra4
}

# ---- [1] 主判据(内存链路, 与上轮 head×自适应 62.6 同尺) ----
rm -f "$MEM_STUDY"
for ds in TN3K DDTIz PTz; do
  run_mem "$ds" efix 0 --stack-e 1.0
  run_mem "$ds" e_adapt    1 --e-adapt "$RATIO" --e-ref "$HFREF"                       # 纯自适应γ(免训练主线)
  run_mem "$ds" head_eadapt 0 --head_ckpt "$OLD_HEAD" --head_hidden "$HIDDEN" \
      --k "$K" --retr "$RETR" --e-adapt "$RATIO" --e-ref "$HFREF"                       # 旧头×自适应
  run_mem "$ds" "$TAG2"    1 --head_ckpt "$NEW_HEAD" --head_hidden "$HIDDEN" \
      --k "$K" --retr "$RETR" --e-adapt "$RATIO" --e-ref "$HFREF"                       # 新头×自适应
done
echo "[p2be] paired (内存) $(date)"
python3 freq_adapt_paired.py "$MEM_STUDY" --pair head_eadapt "$TAG2" || echo "[p2be] WARN pairA"
python3 freq_adapt_paired.py "$MEM_STUDY" --pair e_adapt    "$TAG2"  || echo "[p2be] WARN pairB"
python3 freq_adapt_paired.py "$MEM_STUDY" --pair efix       "$TAG2"  || echo "[p2be] WARN pairC"

# ---- [2] 佐证(磁盘口径, 与 hfal_head 64.0~64.1 同尺) ----
rm -f "$DISK_STUDY"
run_disk DDTIz _hf_g055 hfal_head2
run_disk PTz   _hf_g065 hfal_head2
echo "[p2be] 磁盘口径(新头×每域最优E):"
python3 - "$DISK_STUDY" <<'PY' || echo "[p2be] WARN disk"
import sys, json, numpy as np
vals = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("tag") == "hfal_head2":
        vals[r["ds"]] = float(np.mean(r["per"]["rag_dice"]))
for ds in ["DDTIz", "PTz"]:
    print("  %-7s dice=%.2f" % (ds, vals.get(ds, float("nan"))))
if len(vals) == 2:
    x = (vals["DDTIz"] + vals["PTz"]) / 2.0
    print("  磁盘口径 head×E 跨域=%.2f  vs 旧头 hfal_head 64.02  (Δ=%+.2f)" % (x, x - 64.02))
print("  参照: 旧头×磁盘E hfal_head 跨域=64.02 (DDTI 64.21 / PT 63.83)")
PY

# ---- [3] 刷表 ----
python3 make_crossdomain_table.py
python3 make_bl_comparison.py
echo "[p2be] ALL DONE $(date)"
