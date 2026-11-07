#!/usr/bin/env bash
# =============================================================================
# phase2b_watch.sh — 等 run_phase2b.sh 训练链 [p2b] ALL DONE 且 GPU 空闲后，
#   触发 run_phase2b_eval.sh（修正正则 + 补 e_adapt 纯基线行 + 磁盘口径佐证）。
#   为什么要等：run_phase2b.sh 正在占用 GPU 做 50 轮训练，其自身 [3] 评测矩阵会先跑；
#   本 watcher 只在其结束后追加扩充评测，二者写同一 phase2b_results.jsonl(upsert 幂等)。
#   用法：setsid nohup bash phase2b_watch.sh > phase2b_watch.log 2>&1 &
# =============================================================================
set -u
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1
CHAIN="phase2b_chain.log"

echo "[watch] START $(date)  waiting for [p2b] ALL DONE in $CHAIN"
while ! grep -q "\[p2b\] ALL DONE" "$CHAIN" 2>/dev/null; do sleep 60; done
# 等训练/评测前台进程彻底退出，显存不竞争
while pgrep -f "train.py --modelname SAMUS --task TN3K --epochs 50 --out_tag rag50_e_tr|rag_segment.py --task .* --split test" >/dev/null 2>&1; do
  sleep 20
done
echo "[watch] run_phase2b.sh 链已完成且 GPU 空闲 $(date)"
bash run_phase2b_eval.sh
echo "[watch] ALL DONE $(date)"
