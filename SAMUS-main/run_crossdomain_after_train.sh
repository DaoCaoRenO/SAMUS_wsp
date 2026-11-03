#!/usr/bin/env bash
# 后台守护：等 base50→freq50 链式训练全部结束后，自动对 3 个模型 × 3 个数据集跑跨域测试并汇总。
# 用法：setsid nohup bash run_crossdomain_after_train.sh > crossdomain_watcher.log 2>&1 &
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

CHAIN="train_chain.log"
JSONL="crossdomain_results.jsonl"
BS=8
: > "$JSONL"   # 每次重跑清空旧结果

echo "[watcher] start $(date)"

# 1) 等待链式训练打印 [ALL DONE]（即便某轮训练失败，链脚本也会走到 ALL DONE）
while ! grep -q "\[ALL DONE\]" "$CHAIN" 2>/dev/null; do
  sleep 60
done
echo "[watcher] 训练链已完成，开始测试 $(date)"

# 2) 等待 GPU 上的训练进程真正退出（避免显存竞争）
sleep 20

# 3) 定义模型 tag -> checkpoint
declare -A CK
CK[base50]="checkpoints/TN3K_base50/SAMUS_best.pth"
CK[freq50]="checkpoints/TN3K_freq50/SAMUS_best.pth"
CK[ep270_300]="checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth"

run_test() {
  local tag="$1" ds="$2" ckpt="$3"
  if [ ! -f "$ckpt" ]; then echo "[skip] 缺权重 $tag $ds ($ckpt)"; return; fi
  echo "[run] $tag on $ds ($(basename "$ckpt")) ... $(date)"
  local out
  out=$(python test.py --modelname SAMUS --task "$ds" --load_path "$ckpt" --batch_size "$BS" 2>/dev/null \
        | tr '\r' '\n' | grep '^RESULT_JSON ' | head -1 | sed 's/^RESULT_JSON //')
  if [ -n "$out" ]; then
    echo "$out" | python3 -c "import sys,json;d=json.load(sys.stdin);d['tag']='$tag';d['ds']='$ds';print(json.dumps(d))" >> "$JSONL"
    echo "  -> $(echo "$out" | python3 -c "import sys,json;print('dice=%.2f'%json.load(sys.stdin)['dice'])")"
  else
    echo "  [FAIL] $tag $ds 未解析到结果"
  fi
}

for tag in base50 freq50 ep270_300; do
  for ds in TN3K DDTIz PTz; do
    run_test "$tag" "$ds" "${CK[$tag]}"
  done
done

# 4) 汇总表
python3 make_crossdomain_table.py
echo "[watcher] ALL DONE $(date)"
