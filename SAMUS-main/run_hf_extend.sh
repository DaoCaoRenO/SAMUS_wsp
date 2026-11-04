#!/usr/bin/env bash
# =============================================================================
# 方案E 延伸：(1) 向更低 gain 扩展扫描找最优(0.60 是边界且趋势单调)
#             (2) 记录每张图的 per-image dice 以便做【配对检验】压方差
# 复用已就绪的 ep270 权重与 _hf 数据管线，zero-training。
# 用法：setsid nohup bash run_hf_extend.sh > hf_extend.log 2>&1 &
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

EP270="checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth"
BS=8
HFJSONL="hf_results.jsonl"

emit() {
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

echo "[hf-extend] START $(date)"
# 向更低 gain 扩展（0.60 已是上一轮边界且单调，最优点可能在更低处）
for g in 0.35 0.45 0.55; do
  echo "[E+] build gain=$g $(date)"
  python hf_boost.py --gain "$g" --ratio 0.25
  for t in TN3K_hf DDTIz_hf PTz_hf; do
    echo "[E+] test $t gain=$g $(date)"
    out=$(python test.py --modelname SAMUS --task "$t" --load_path "$EP270" --batch_size "$BS" 2>/dev/null \
          | tr '\r' '\n' | grep '^RESULT_JSON ' | head -1 | sed 's/^RESULT_JSON //')
    if [ -n "$out" ]; then
      echo "$out" | emit "$HFJSONL" "gain=$g" "ds=$t"
      echo "   -> dice=$(echo "$out" | python3 -c 'import sys,json;print("%.2f"%json.load(sys.stdin)["dice"])')"
    else
      echo "   [FAIL] $t gain=$g"
    fi
  done
done
python3 make_hf_table.py
echo "[hf-extend] ALL DONE $(date)"
