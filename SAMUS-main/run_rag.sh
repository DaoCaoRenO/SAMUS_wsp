#!/usr/bin/env bash
# =============================================================================
# run_rag.sh — 原型+RAG 检索增强(免训练) 全流程编排（对应计划 §6）
#   链路：build 记忆库 → val 网格锁参(防泄漏) → 用锁定参一次性跑 DDTIz/PTz(写总表)
#         → 消融(写 rag_results.jsonl) → 刷新跨域总表 + 跨口径对比总表。
#   用法：bash run_rag.sh            # 全量(会重算记忆库与选参)
#         SKIP_BUILD=1 bash run_rag.sh   # 已有 rag_bank/tn3k_bank.pt 时跳过建库
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

CKPT="checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth"
BANK="rag_bank/tn3k_bank.pt"
VALBEST="rag_valbest.json"
TABLE="crossdomain_results.jsonl"
LOG="run_rag.log"
exec > >(tee -a "$LOG") 2>&1
echo "==== run_rag $(date '+%F %T') ===="

# 1) 构建并缓存 TN3K train 记忆库
if [[ "${SKIP_BUILD:-0}" != "1" || ! -f "$BANK" ]]; then
  echo "--- [1] build_bank ---"
  python3 rag_build_bank.py --ckpt "$CKPT" --split train --out "$BANK"
else
  echo "--- [1] build_bank SKIPPED (cached $BANK) ---"
fi

# 2) 仅在源域 TN3K val 上网格锁参（绝不碰 DDTIz/PTz 测试集调 λ）
echo "--- [2] select (TN3K val grid k×λ×τ) ---"
python3 rag_select.py --ckpt "$CKPT" --bank "$BANK" --out "$VALBEST"
read -r K LAM TAU < <(python3 -c "import json;d=json.load(open('$VALBEST'))['best'];print(d['k'],d['lambda'],d['tau'])")
echo "--- locked: k=$K lambda=$LAM tau=$TAU ---"

# 3) 用锁定参一次性跑跨域测试（写 rag_results.jsonl + upsert 总表 rag 行）
echo "--- [3] final cross-domain eval (locked) ---"
rm -f rag_results.jsonl           # 干净的研究日志
for T in DDTIz PTz; do
  echo "== $T =="
  python3 rag_segment.py --task "$T" --split test \
    --k "$K" --lambda "$LAM" --tau "$TAU" --retr roi --agg softmax \
    --out rag_results.jsonl --emit-table "$TABLE"
done

# 4) 消融（只进研究日志 rag_results.jsonl，不进总表）
echo "--- [4] ablations ---"
for T in DDTIz PTz; do
  for KK in 1 5 10; do
    python3 rag_segment.py --task "$T" --split test --k "$KK" --lambda "$LAM" --tau "$TAU" --out rag_results.jsonl
  done
  python3 rag_segment.py --task "$T" --split test --k "$K" --lambda "$LAM" --tau "$TAU" --retr mean --out rag_results.jsonl   # ROI vs 均值检索
  python3 rag_segment.py --task "$T" --split test --k 5 --lambda "$LAM" --tau "$TAU" --agg top1 --out rag_results.jsonl         # 加权聚合 vs 固定top1
  python3 rag_segment.py --task "$T" --split test --k "$K" --lambda "$LAM" --tau "$TAU" --stack-e 1.5 --out rag_results.jsonl   # 是否叠加方案E
done

# 5) 刷新两张总表
echo "--- [5] refresh tables ---"
python3 make_crossdomain_table.py
python3 make_bl_comparison.py
echo "==== run_rag done ===="
