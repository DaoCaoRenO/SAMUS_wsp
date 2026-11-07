#!/usr/bin/env bash
# =============================================================================
# run_phase2b.sh — Phase2B（§38 门槛对齐后已满足）：PCR 头在 **E 增强后的 base** 上重训
#   为什么要重训：现用的 TN3K_rag50 头是在**未经 E 增强的 base** 上学的，而评测时 E
#   已经先把目标域高频压回源域量级 → 头见到的是「E 之后的图」，训练/评测口径不一致。
#   严格版 Phase2B = 训练时也对 imgs_q 施加同一套 E(此处用逐图自适应γ，与 eval 的
#   --e-adapt 完全同式：gain_i=clamp(E_src_ref/hf_i, lo, hi))，让头学「E 之后还剩什么错」。
#   口径纪律(承 2026-11-07)：训练用的 hfa_boost 是**内存版**，所以主判据必须也走内存链路
#   (--e-adapt 评测)，与上一轮 head×自适应(tag=head_eadapt, 跨域 62.3) 严格同尺比；
#   磁盘口径(hfal)只作方向性佐证，不与内存数值混比。
#   链路：冒烟 1 轮(护栏 val dice≈0.83 + 确认 e-adapt 生效) → 50 轮训练 → 三集评测
#         → 跨行同图配对(old head vs new head) → 刷两表。
#   用法：setsid nohup bash run_phase2b.sh > phase2b_chain.log 2>&1 &
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

CKPT="checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth"
BANK="rag_bank/tn3k_bank.pt"
OLD_HEAD="checkpoints/TN3K_rag50/SAMUS_best.pth"
HIDDEN=32; K=5; RETR=roi; SHIFT=1; SHIFT_STR=1.0
RATIO=0.25                        # 自适应γ 高频半径(与 run_freq_adapt.sh/磁盘口径一致)
HFREF="rag_bank/hf_ref.json"
E_REF="$(python3 -c "import json;print(json.load(open('$HFREF'))['median'])" 2>/dev/null || echo 0)"
CDJSONL="crossdomain_results.jsonl"
STUDY="phase2b_results.jsonl"
TAG2=raghead_e_tr2                # 新头×自适应γ(与研究日志并列, 不覆上轮 head_eadapt)

[ -f "$HFREF" ] || { echo "[p2b] 缺 $HFREF(先跑 run_freq_adapt.sh 生成 E_src_ref)，终止"; exit 1; }
echo "[p2b] START $(date)  E_src_ref=$E_REF ratio=$RATIO head(base未E增强)=$OLD_HEAD"

RAGARGS="--rag_head 1 --rag_head_hidden $HIDDEN --rag_k $K --rag_retr $RETR --rag_shift $SHIFT \
         --rag_shift_strength $SHIFT_STR --rag_bank $BANK --rag_freeze_all 1 --rag_base_ckpt $CKPT \
         --rag_stack_e_adapt $RATIO --rag_e_ref $E_REF"

# ---- [1] 冒烟 1 轮：护栏 = val dice≈0.83(零初始化头恒等 base) + e-adapt 分支被走到 ----
echo "[p2b] smoke 1 epoch $(date)"
python train.py --modelname SAMUS --task TN3K --epochs 1 --out_tag rag2b_smoke $RAGARGS \
  > train_rag2b_smoke.log 2>&1
grep -q "val dice" train_rag2b_smoke.log || { echo "[p2b] 冒烟失败(无 val dice)"; tail -30 train_rag2b_smoke.log; exit 1; }
SMOKE_DICE=$(grep -o "val dice [0-9.]*" train_rag2b_smoke.log | tail -1 | awk '{print $3}')
echo "[p2b] smoke val dice=$SMOKE_DICE"
python3 - "$SMOKE_DICE" <<'PY' || echo "[p2b] WARN 护栏判定"
import sys
d = float(sys.argv[1])
print("  [护栏] smoke val dice=%.4f %s (零初始化头应≈base 0.835)" % (d, "OK" if d > 0.80 else "偏低!查 E 增强是否破坏训练"))
PY
NPCR=$(python3 -c "import torch;sd=torch.load('checkpoints/TN3K_rag2b_smoke/SAMUS_best.pth',map_location='cpu');print(sum(1 for k in sd if k.startswith('pcr.')))" 2>/dev/null || echo 0)
[ "$NPCR" -ge 1 ] || { echo "[p2b] 冒烟权重无 pcr.* 键(NPCR=$NPCR)，终止"; exit 1; }

# ---- [2] 50 轮正式训练 ----
echo "[p2b] train 50 epochs $(date)"
python train.py --modelname SAMUS --task TN3K --epochs 50 --out_tag rag50_e_tr $RAGARGS \
  > train_rag50_e_tr.log 2>&1
echo "[p2b] train rc=$? $(date)"
NEW_HEAD="checkpoints/TN3K_rag50_e_tr/SAMUS_best.pth"
[ -f "$NEW_HEAD" ] || { echo "[p2b] 缺 $NEW_HEAD，训练未产出权重，终止"; exit 1; }

# ---- [3] 评测矩阵(内存链路, 与上轮同尺) ----
rm -f "$STUDY"

run() {  # run <ds> <tag> <emit> [extra args...]
  local ds="$1" tag="$2" emit="$3"; shift 3
  local extra=()
  [ "$emit" = "1" ] && extra=(--emit-table "$CDJSONL")
  echo "==== ds=$ds tag=$tag $* ===="
  python3 rag_segment.py --task "$ds" --split test --ckpt "$CKPT" --bank "$BANK" \
    --lambda 0 --batch 8 --out "$STUDY" --tag "$tag" "${extra[@]}" "$@"
}

for ds in TN3K DDTIz PTz; do
  run "$ds" efix     0 --stack-e 1.0                                             # 同链路 γ=1 对照
  run "$ds" head_eadapt     0 --head_ckpt "$OLD_HEAD" --head_hidden "$HIDDEN" \
      --k "$K" --retr "$RETR" --e-adapt "$RATIO" --e-ref "$HFREF"               # 旧头×自适应(上轮)
  run "$ds" "$TAG2"  1 --head_ckpt "$NEW_HEAD" --head_hidden "$HIDDEN" \
      --k "$K" --retr "$RETR" --e-adapt "$RATIO" --e-ref "$HFREF"               # 新头×自适应
done

# ---- [4] 跨行同图配对：重训是否把 head 的边际做大 ----
echo "[p2b] paired analysis $(date)"
python3 freq_adapt_paired.py "$STUDY" --pair head_eadapt "$TAG2" || echo "[p2b] WARN pairA"
python3 freq_adapt_paired.py "$STUDY" --pair efix "$TAG2"        || echo "[p2b] WARN pairB"

# ---- [5] 刷两张总表 ----
python3 make_crossdomain_table.py
python3 make_bl_comparison.py
echo "[p2b] ALL DONE $(date)"
