#!/usr/bin/env bash
# =============================================================================
# 叠加测试：edge50(边缘增强第1层) 权重 × 方案E(测试时高频衰减)
#   验证"区域(E涨dice) + 边界(edge降HD95)"能否同时拿下。纯测试、零训练。
#   对 gain ∈ {0.60, 0.65}(方案E实测峰区) 分别 build _hf 数据 -> 用 edge50 测 3 集。
#   edge50 无 Freq_Adapter，故 --freq_adapter 0(strict 加载)。
# 用法：setsid nohup bash run_edge_hf.sh > edge_hf.log 2>&1 &
# =============================================================================
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

CKPT="checkpoints/TN3K_edge50/SAMUS_best.pth"
BS=8
OUT="edge_hf_results.jsonl"

echo "[edge+hf] START $(date)  ckpt=$CKPT"
if [ ! -f "$CKPT" ]; then echo "[edge+hf] 缺权重 $CKPT，终止"; exit 1; fi
rm -f "$OUT"

for g in 0.60 0.65; do
  echo "[edge+hf] build gain=$g $(date)"
  python hf_boost.py --gain "$g" --ratio 0.25
  for t in TN3K_hf DDTIz_hf PTz_hf; do
    echo "[edge+hf] test $t gain=$g $(date)"
    out=$(python test.py --modelname SAMUS --task "$t" --load_path "$CKPT" --batch_size "$BS" 2>/dev/null \
          | tr '\r' '\n' | grep '^RESULT_JSON ' | head -1 | sed 's/^RESULT_JSON //')
    if [ -n "$out" ]; then
      echo "$out" | GAIN="$g" DS="$t" python3 -c "
import sys,json,os
d=json.load(sys.stdin); d['gain']=float(os.environ['GAIN']); d['ds']=os.environ['DS']; d['model']='edge50'
print(json.dumps(d))" >> "$OUT"
      echo "   -> dice=$(echo "$out" | python3 -c 'import sys,json;print("%.2f"%json.load(sys.stdin)["dice"])') HD95=$(echo "$out" | python3 -c 'import sys,json;print("%.2f"%json.load(sys.stdin)["hdis"])')"
    else
      echo "   [FAIL] $t gain=$g"
    fi
  done
done

# 汇总对比：edge50(无E) vs edge50+E vs ep270+E
python3 - "$OUT" <<'PY'
import json,sys
rows={}
for l in open(sys.argv[1]):
    d=json.loads(l); rows[(round(d['gain'],2),d['ds'])]=d
# edge50 无E(来自 crossdomain_results.jsonl)
base={}
for l in open('crossdomain_results.jsonl'):
    d=json.loads(l)
    if d.get('tag')=='edge50': base[d['ds']]=d
    if d.get('tag')=='ep270_300': base['ep270_'+d['ds']]=d
DS=[('TN3K','TN3K_hf'),('DDTIz','DDTIz_hf'),('PTz','PTz_hf')]
print('\n==== 叠加测试汇总 ====')
print('%-8s %-14s %-14s %-14s'%('数据集','edge50(无E)','edge50+E g.60','edge50+E g.65'))
for raw,hf in DS:
    b=base.get(raw,{})
    c60=rows.get((0.6,hf),{}); c65=rows.get((0.65,hf),{})
    print('%-8s dice%5.1f/hd%5.1f  dice%5.1f/hd%5.1f  dice%5.1f/hd%5.1f'%(
        raw, b.get('dice',0),b.get('hdis',0), c60.get('dice',0),c60.get('hdis',0), c65.get('dice',0),c65.get('hdis',0)))
# 跨域均值
for lbl,sel in [('edge50(无E)',lambda hf: base.get({'TN3K_hf':'TN3K','DDTIz_hf':'DDTIz','PTz_hf':'PTz'}[hf],{})),
                ('edge50+E.60',lambda hf: rows.get((0.6,hf),{})),
                ('edge50+E.65',lambda hf: rows.get((0.65,hf),{}))]:
    dd=sel('DDTIz_hf').get('dice',0); pp=sel('PTz_hf').get('dice',0)
    hd=(sel('DDTIz_hf').get('hdis',0)+sel('PTz_hf').get('hdis',0))/2
    print('  %-12s 跨域均值 dice=%.2f  HD95=%.2f'%(lbl,(dd+pp)/2,hd))
PY
echo "[edge+hf] ALL DONE $(date)"
