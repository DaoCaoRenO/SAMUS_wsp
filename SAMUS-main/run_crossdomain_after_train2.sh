#!/usr/bin/env bash
# 守护 v2：等 fad50→fadaug50 链式训练结束后，自动跑 2 模型 × 3 数据集跨域测试，
# 结果【追加】进 crossdomain_results.jsonl（保留 base50/freq50/ep270_300 已有 9 行），再重生成 2×2 消融表。
# 用法：setsid nohup bash run_crossdomain_after_train2.sh > crossdomain_watcher2.log 2>&1 &
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

CHAIN="train_chain2.log"
JSONL="crossdomain_results.jsonl"
BS=8

echo "[watcher2] start $(date)"

while ! grep -q "\[ALL DONE\]" "$CHAIN" 2>/dev/null; do
  sleep 60
done
echo "[watcher2] 训练链完成，开始测试 $(date)"
sleep 20   # 等训练进程真正退出，避免显存竞争

declare -A CK
CK[fad50]="checkpoints/TN3K_fad50/SAMUS_best.pth"
CK[fadaug50]="checkpoints/TN3K_fadaug50/SAMUS_best.pth"

run_test() {
  local tag="$1" ds="$2" ckpt="$3"
  if [ ! -f "$ckpt" ]; then echo "[skip] 缺权重 $tag $ds ($ckpt)"; return; fi
  echo "[run] $tag on $ds ... $(date)"
  local out
  out=$(python test.py --modelname SAMUS --task "$ds" --load_path "$ckpt" \
          --batch_size "$BS" --freq_adapter 1 2>/dev/null \
        | tr '\r' '\n' | grep '^RESULT_JSON ' | head -1 | sed 's/^RESULT_JSON //')
  if [ -n "$out" ]; then
    echo "$out" | python3 -c "import sys,json;d=json.load(sys.stdin);d['tag']='$tag';d['ds']='$ds';print(json.dumps(d))" >> "$JSONL"
    echo "  -> $(echo "$out" | python3 -c "import sys,json;print('dice=%.2f'%json.load(sys.stdin)['dice'])")"
  else
    echo "  [FAIL] $tag $ds 未解析到结果"
  fi
}

for tag in fad50 fadaug50; do
  for ds in TN3K DDTIz PTz; do
    run_test "$tag" "$ds" "${CK[$tag]}"
  done
done

python3 make_crossdomain_table.py
echo "[watcher2] ALL DONE $(date)"
