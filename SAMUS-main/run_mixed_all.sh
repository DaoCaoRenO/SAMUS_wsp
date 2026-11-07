#!/usr/bin/env bash
# =============================================================================
# run_mixed_all.sh — 混合训练对照：TN3K + DDTI(剔47测试) + PT(剔44测试) 全参微调 50 轮
#   问题：之前全程 source-only 零样本(ep270 DDTI 58.3 / base50 DDTI 60.3)。若把目标域
#         标注数据一起放进训练，DDTI/PT 能涨到哪 —— 量化"目标域数据可用"的上界。
#   数据：prep_mixed_data.py 产出 train_allsrc.txt(2803=TN3K2000+DDTI_tr419+PT_tr384)，
#         已剔除 DDTIz/PTz 测试子集 → 训练∩测试=∅，评测仍用 DDTIz(47)/PTz(44)/TN3K(379) 同尺。
#   对照：mixedall vs base50（同为 50 轮、同 SAM 初始化、全参、lr1e-4/bs8；唯一变量=加了 DDTI/PT）
#         → rag_segment --lambda 0 产 per-image，freq_adapt_paired --pair 出同图 Wilcoxon ΔDice。
#   链路：冒烟1轮(护栏 new best dice>0.5) → 50轮训练 → 双模型三集评测 → paired → 刷两表。
#   用法：setsid nohup bash run_mixed_all.sh > mixed_chain.log 2>&1 &
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

BANK="rag_bank/tn3k_bank.pt"
BASE50="checkpoints/TN3K_base50/SAMUS_best.pth"          # 匹配对照(50轮TN3K-only)
SPLITFILE="dataset/SAMUS/MainPatient/train_allsrc.txt"
CDJSONL="crossdomain_results.jsonl"
STUDY="mixed_study.jsonl"
BS=8

[ -f "$SPLITFILE" ] || { echo "[mix] 缺 $SPLITFILE，先跑 prep_mixed_data.py，终止"; exit 1; }
[ -f "$BASE50" ]   || { echo "[mix] 缺对照权重 $BASE50，终止"; exit 1; }
N=$(grep -c . "$SPLITFILE")
echo "[mix] START $(date)  train_allsrc=$N 行  对照=$BASE50"

# ---- [1] 冒烟 1 轮：护栏 = 管线跑通(有 new best dice 行) + best ckpt 落盘 ----
#   注：全参从 SAM ViT-B 随机 decoder 起步，epoch0 val dice≈0 属正常(base50/ep270 当初同为0)，
#   故此处不卡收敛阈值，仅验证数据/前向/反向/评测/存盘链路无异常。
echo "[mix] smoke 1 epoch $(date)"
python train.py --modelname SAMUS --task MixedAll --epochs 1 --out_tag mix_smoke \
  > train_mix_smoke.log 2>&1
SMOKE_DICE=$(grep -o "new best dice [0-9.]*" train_mix_smoke.log | tail -1 | awk '{print $4}')
SMOKE_CKPT="checkpoints/MixedAll_mix_smoke/SAMUS_best.pth"
echo "[mix] smoke new best dice=${SMOKE_DICE:-NA}  ckpt=$SMOKE_CKPT"
[ -n "$SMOKE_DICE" ] || { echo "[mix] 冒烟失败(无 new best dice 行)，终止"; tail -25 train_mix_smoke.log; exit 1; }
[ -f "$SMOKE_CKPT" ]  || { echo "[mix] 冒烟未产出 best ckpt，终止"; tail -25 train_mix_smoke.log; exit 1; }
python3 -c "import math;d=float('$SMOKE_DICE');print('  [护栏] smoke dice=%.4f 非NaN OK (从头全参1轮偏低属正常)'%d) if not math.isnan(d) else (_ for _ in ()).throw(ValueError('NaN dice'))" \
  || { echo "[mix] dice 为 NaN，终止"; exit 1; }

# ---- [2] 50 轮正式全参混合训练 ----
echo "[mix] train 50 epochs $(date)"
python train.py --modelname SAMUS --task MixedAll --epochs 50 --out_tag a50 \
  > train_mix_a50.log 2>&1
echo "[mix] train rc=$? $(date)"
MIXED="checkpoints/MixedAll_a50/SAMUS_best.pth"
[ -f "$MIXED" ] || { echo "[mix] 缺 $MIXED，训练未产出，终止"; exit 1; }
BEST=$(grep -o "new best dice [0-9.]*" train_mix_a50.log | tail -1)
echo "[mix] train done: $BEST"

# ---- [3] 评测：base50(对照) 与 mixedall(实验) 三集 λ=0 同图 ----
rm -f "$STUDY"
run_eval() {  # run_eval <ds> <tag> <ckpt> <emit1/0>
  local ds="$1" tag="$2" ckpt="$3" emit="$4"
  local extra=(); [ "$emit" = "1" ] && extra=(--emit-table "$CDJSONL")
  echo "==== [eval] ds=$ds tag=$tag ckpt=$(basename "$(dirname "$ckpt")") ===="
  python3 rag_segment.py --task "$ds" --split test --ckpt "$ckpt" --bank "$BANK" \
    --lambda 0 --batch "$BS" --out "$STUDY" --tag "$tag" "${extra[@]}"
}
for ds in TN3K DDTIz PTz; do
  run_eval "$ds" mixed_base50 "$BASE50" 0
  run_eval "$ds" mixedall     "$MIXED"  1
done

# ---- [4] 同图配对：混训相对 base50 的 ΔDice ----
echo "[mix] paired (mixed_base50 -> mixedall) $(date)"
python3 freq_adapt_paired.py "$STUDY" --pair mixed_base50 mixedall || echo "[mix] WARN pair"

# ---- [5] 刷两表 ----
python3 make_crossdomain_table.py
python3 make_bl_comparison.py
echo "[mix] ALL DONE $(date)"
