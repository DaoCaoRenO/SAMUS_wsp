#!/usr/bin/env bash
# =============================================================================
# run_train_shift.sh — Phase3B 全量合成域偏移训练 (对症 zero-shot 对照, 计划 §3B)
#   假设：zero-shot 跨域失败根因是主干从未见过「跨域感」图；在普通训练分支对每张
#   训练图施加 R.shift_image(合成域偏移)，让主干也学到退化不变性。
#   严格镜像 base50，只加 --train_shift 1 --train_shift_strength 1.0 (§50)。
#   对照：shift50 vs base50 三集 (§52；test.py 口径绝对值入总表, paired 见评测说明)。
#   链路：冒烟1轮(校验不崩+val dice) → 正式50轮(setsid nohup 脱离 IDE cgroup) →
#         三集评测(tag=shift50) → 刷两张总表。
#   用法：setsid nohup bash run_train_shift.sh > train_shift_chain.log 2>&1 &
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

BS=8
STRENGTH=1.0
CDJSONL="crossdomain_results.jsonl"
SHIFT_CKPT="checkpoints/TN3K_shift50/SAMUS_best.pth"

echo "[shift] START $(date)  (train_shift=1 strength=$STRENGTH)"

# ---- [1] 冒烟 1 轮：不崩 + 出 val dice(护栏，避免 shift 过强训崩) ----
python train.py --modelname SAMUS --task TN3K --epochs 1 --out_tag shift_smoke \
    --train_shift 1 --train_shift_strength "$STRENGTH" > train_shift_smoke.log 2>&1
if ! grep -q "val dice" train_shift_smoke.log; then
  echo "[shift] 冒烟失败(无 val dice)，终止。尾部："; tail -30 train_shift_smoke.log; exit 1
fi
echo "[shift] 冒烟通过 $(date)"; grep -E "val dice|Total_params" train_shift_smoke.log | tail -4

# ---- [2] 正式 50 轮 (严格镜像 base50，仅加开关) ----
echo "[shift] 正式训练 START $(date)"
python train.py --modelname SAMUS --task TN3K --epochs 50 --out_tag shift50 \
    --train_shift 1 --train_shift_strength "$STRENGTH" > train_shift50.log 2>&1
echo "[shift] 训练完成 rc=$? $(date)"

# ---- [3] 三集评测 (test.py RESULT_JSON -> 总表 tag=shift50) ----
run_test() {
  local tag="$1" ds="$2" ckpt="$3"
  if [ ! -f "$ckpt" ]; then echo "[skip] 缺权重 $tag $ds ($ckpt)"; return; fi
  echo "[run] $tag on $ds ($(basename "$ckpt")) ... $(date)"
  local out
  out=$(python test.py --modelname SAMUS --task "$ds" --load_path "$ckpt" --batch_size "$BS" 2>/dev/null \
        | tr '\r' '\n' | grep '^RESULT_JSON ' | head -1 | sed 's/^RESULT_JSON //')
  if [ -n "$out" ]; then
    echo "$out" | python3 -c "import sys,json;d=json.load(sys.stdin);d['tag']='$tag';d['ds']='$ds';print(json.dumps(d))" >> "$CDJSONL"
    echo "  -> $(echo "$out" | python3 -c "import sys,json;print('dice=%.2f'%json.load(sys.stdin)['dice'])")"
  else
    echo "  [FAIL] $tag $ds 未解析到结果"
  fi
}
if [ -f "$SHIFT_CKPT" ]; then
  for ds in TN3K DDTIz PTz; do run_test shift50 "$ds" "$SHIFT_CKPT"; done
else
  echo "[shift] 缺权重 $SHIFT_CKPT，跳过评测"
fi

# ---- [4] 刷两张总表 ----
echo "[shift] refresh tables $(date)"
python3 make_crossdomain_table.py
python3 make_bl_comparison.py
echo "[shift] ALL DONE $(date)"
