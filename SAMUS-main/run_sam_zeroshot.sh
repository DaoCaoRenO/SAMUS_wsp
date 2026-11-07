#!/usr/bin/env bash
# =============================================================================
# run_sam_zeroshot.sh — 归因消融：完全不训 TN3K 的原生 SAM + 原型/RAG(TN3K train 仅当检索库)
#   问题：现有零样本成绩(ep270 58.3 / base50 60.3 / 方案E 63.6)都内嵌了「TN3K 微调过的
#         encoder」。把 encoder 换成未微调的原生 SAM ViT-B、只保留「原型/RAG 机制」，
#         跨域还剩多少 → 与微调链之差 ≈ TN3K 训练买到的 dice(训练贡献)。
#   口径：真·零训练(无任何梯度)；源域 TN3K train 仅出现在推理侧建原型库，与 rag_bank
#         零样本口径一致；评测复用 rag_common.make_loader + seg_metrics(与 base50 同尺)。
#   两模式：map=稠密原型最近邻(源域 val 标定 thr)；amg=原生 SAM AMG 候选+原型选掩码。
#   链路：建全量库(2000) → base50 参照 per-image(λ=0) → 三集 map+amg(emit 总表)
#         → paired(base50 vs map/amg 同图 Wilcoxon=训练贡献) → 刷两表。
#   用法：setsid nohup bash run_sam_zeroshot.sh > samzs_chain.log 2>&1 &
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

BANK="rag_bank/sam_tn3k_bank.pt"
BASE50="checkpoints/TN3K_base50/SAMUS_best.pth"          # 微调参照(同 loader 同尺)
SAMUS_BANK="rag_bank/tn3k_bank.pt"                        # rag_segment 需给库(λ=0 不影响输出)
CDJSONL="crossdomain_results.jsonl"
STUDY="samzs_study.jsonl"
K=5

[ -f "$BASE50" ] || { echo "[zs] 缺参照权重 $BASE50，终止"; exit 1; }
echo "[zs] START $(date)  bank=$BANK  参照=$BASE50"

# ---- [1] 建全量原生-SAM 原型库(TN3K train 2000)：一次性，护栏=文件生成 ----
echo "[zs] build full bank (TN3K train) $(date)"
python sam_zeroshot.py --build-bank --bank "$BANK" --no-eval > samzs_bank.log 2>&1
[ -f "$BANK" ] || { echo "[zs] 建库失败，终止"; tail -25 samzs_bank.log; exit 1; }
grep -q "\[bank\] built" samzs_bank.log || { echo "[zs] 建库日志异常，终止"; tail -25 samzs_bank.log; exit 1; }
echo "[zs] bank ok: $(grep '\[bank\] built' samzs_bank.log | tail -1)"

# ---- [2] base50(微调参照) per-image 写进同一 STUDY，供同图配对(λ=0) ----
rm -f "$STUDY"
for ds in TN3K DDTIz PTz; do
  echo "==== [ref] rag_segment base50 λ=0 ds=$ds $(date) ===="
  python3 rag_segment.py --task "$ds" --split test --ckpt "$BASE50" --bank "$SAMUS_BANK" \
    --lambda 0 --batch 8 --out "$STUDY" --tag samzs_base50ref || echo "[zs] WARN ref $ds"
done

# ---- [3] 原生 SAM + 原型：map(源域 val 标定 thr) 与 amg 两模式，三集，emit 总表 ----
for ds in TN3K DDTIz PTz; do
  echo "==== [zs] map ds=$ds $(date) ===="
  python sam_zeroshot.py --bank "$BANK" --task "$ds" --split test --mode map --k "$K" \
    --calibrate --out "$STUDY" --emit-table "$CDJSONL" || echo "[zs] WARN map $ds"
  echo "==== [zs] amg ds=$ds $(date) ===="
  python sam_zeroshot.py --bank "$BANK" --task "$ds" --split test --mode amg --k "$K" \
    --out "$STUDY" --emit-table "$CDJSONL" || echo "[zs] WARN amg $ds"
done

# ---- [4] 同图配对：训练贡献 = raw SAM+原型 相对微调 base50 差多少 ----
echo "[zs] paired (base50 -> samzs) $(date)"
python3 freq_adapt_paired.py "$STUDY" --pair samzs_base50ref samzs_map || echo "[zs] WARN pair map"
python3 freq_adapt_paired.py "$STUDY" --pair samzs_base50ref samzs_amg || echo "[zs] WARN pair amg"

# ---- [5] 刷两表 ----
python3 make_crossdomain_table.py
python3 make_bl_comparison.py
echo "[zs] ALL DONE $(date)"
