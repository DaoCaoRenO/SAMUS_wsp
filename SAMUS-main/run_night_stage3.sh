#!/usr/bin/env bash
# =============================================================================
# 夜间自主编排 Stage3+Stage4（目标：努力用频域技术涨点）
#   前置：run_fad_chain.sh(方案C训练) + run_crossdomain_after_train2.sh(守护v2测试) 已在跑。
#   本脚本：
#     Stage-E  等守护v2完成 -> 测试时高频幅度增益扫描（zero-training，~15min）
#     Stage-D  自适应决策 -> 冒烟1轮 -> FFL正式50轮 -> 自动跨域测试 -> 更新汇总表
#   用法：setsid nohup bash run_night_stage3.sh > night_stage3.log 2>&1 &
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

EP270="checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth"
BS=8
HFJSONL="hf_results.jsonl"
CDJSONL="crossdomain_results.jsonl"

# test.py -> 取 RESULT_JSON，注入额外字段后追加到 jsonl
emit() {  # $1=jsonl $2=key=val(可多个)
  local jsonl="$1"; shift
  python3 -c "
import sys,json
d=json.load(sys.stdin)
for kv in sys.argv[1:]:
    k,v=kv.split('=',1)
    try: v=float(v)
    except: pass
    d[k]=v
print(json.dumps(d))
" "$@" >> "$jsonl"
}

echo "[stage3] START $(date)"

# ---- 等守护v2（方案C测试 + 2×2表）完成 ----
while ! grep -q "\[watcher2\] ALL DONE" crossdomain_watcher2.log 2>/dev/null; do
  sleep 60
done
echo "[stage3] watcher2 完成，进入方案E $(date)"
sleep 20   # 让测试进程完全退出，避免显存竞争

# ================= Stage-E：测试时高频幅度增益扫描 =================
echo "===== [Stage-E] 高频增益扫描 $(date) ====="
rm -f "$HFJSONL"
python hf_boost.py --measure --ratio 0.25 | tee hf_measure.txt

for g in 0.60 0.65 0.75 0.90 1.00 1.20; do
  echo "[E] build gain=$g $(date)"
  python hf_boost.py --gain "$g" --ratio 0.25
  for t in TN3K_hf DDTIz_hf PTz_hf; do
    echo "[E] test $t gain=$g $(date)"
    out=$(python test.py --modelname SAMUS --task "$t" --load_path "$EP270" --batch_size "$BS" 2>/dev/null \
          | tr '\r' '\n' | grep '^RESULT_JSON ' | head -1 | sed 's/^RESULT_JSON //')
    if [ -n "$out" ]; then
      echo "$out" | emit "$HFJSONL" "gain=$g" "ds=$t"
      echo "   -> dice=$(echo "$out" | python3 -c 'import sys,json;print("%.2f"%json.load(sys.stdin)["dice"])')"
    else
      echo "   [FAIL] $t gain=$g 未解析到结果"
    fi
  done
done
python3 make_hf_table.py
echo "[stage3] 方案E完成 $(date)"

# ================= Stage-D：Focal Frequency Loss 训练 =================
echo "===== [Stage-D] 频域损失训练 $(date) ====="
python3 decide_stage4.py
source stage4_flags.sh
echo "[stage3] StageD flags: FA=$FA AUG=$AUG REASON=$REASON"

# 冒烟 1 轮（验证 FFL 不炸、能出 val dice）
python train.py --modelname SAMUS --task TN3K --epochs 1 --out_tag floss_smoke \
       --freq_loss 1 --freq_adapter "$FA" --freq_aug "$AUG" > train_floss_smoke.log 2>&1

if grep -q "val dice" train_floss_smoke.log; then
  echo "[stage3] 冒烟通过 -> 正式50轮 $(date)"
  grep -E "freq|val dice|loss" train_floss_smoke.log | tail -6

  python train.py --modelname SAMUS --task TN3K --epochs 50 --out_tag floss50 \
         --freq_loss 1 --freq_adapter "$FA" --freq_aug "$AUG" > train_floss50.log 2>&1
  echo "[stage3] floss50 训练完成 rc=$? $(date)"

  CKPT="checkpoints/TN3K_floss50/SAMUS_best.pth"
  if [ -f "$CKPT" ]; then
    for ds in TN3K DDTIz PTz; do
      echo "[D] test floss50 on $ds $(date)"
      out=$(python test.py --modelname SAMUS --task "$ds" --load_path "$CKPT" --batch_size "$BS" \
              --freq_adapter "$FA" 2>/dev/null \
            | tr '\r' '\n' | grep '^RESULT_JSON ' | head -1 | sed 's/^RESULT_JSON //')
      if [ -n "$out" ]; then
        echo "$out" | emit "$CDJSONL" "tag=floss50" "ds=$ds"
        echo "   -> dice=$(echo "$out" | python3 -c 'import sys,json;print("%.2f"%json.load(sys.stdin)["dice"])')"
      else
        echo "   [FAIL] floss50 $ds 未解析到结果"
      fi
    done
    python3 make_crossdomain_table.py
  else
    echo "[stage3] 缺 floss50 权重，跳过测试"
  fi
else
  echo "[stage3] 冒烟失败，跳过正式训练。见 train_floss_smoke.log 尾部："
  tail -20 train_floss_smoke.log
fi

echo "[stage3] ALL DONE $(date)"
