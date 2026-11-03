#!/usr/bin/env python3
"""
夜间 Stage4（方案D：Focal Frequency Loss 训练）自适应决策。

读取 crossdomain_results.jsonl 里方案C的产物（fad50=C单独、fadaug50=A+C），
与对照组 base50 比，算【跨域均值 ΔDice】及合并 SE，据此决定 Stage4 该带哪种结构/增广：

  决策规则（目标=努力用频域技术涨点，只跑得动一次 50 轮，故挑最有希望的组合）：
  - 若 C 或 A+C 有正向趋势(Δ>0)：把方案D(FFL)叠加到更优的那个结构上，
    即 freq_adapter/freq_aug 沿用该配置 —— 让"频域损失 + 频域结构"协同。
  - 若两者都非正：说明频域结构本身无益，改为 D 单独(freq_adapter=0,freq_aug=0)，
    干净地检验"频域监督信号"这条正交路线是否独立涨点。

输出 stage4_flags.sh 供 bash source：FA=0/1, AUG=0/1, REASON="..."
"""
import json, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, "crossdomain_results.jsonl")
OUT = os.path.join(HERE, "stage4_flags.sh")
N = {"TN3K": 379, "DDTIz": 47, "PTz": 44}
CROSS = ["DDTIz", "PTz"]          # 只看跨域两集（域内 TN3K 不是本课题重点）

rows = {}
for line in open(JSONL):
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    rows[(d.get("tag"), d.get("ds"))] = d


def delta_mean(tag):
    """tag 相对 base50 的跨域均值 ΔDice，及合并 SE（逐图 std/√n 合成）。"""
    ds_list, ses = [], []
    for ds in CROSS:
        b = rows.get(("base50", ds))
        t = rows.get((tag, ds))
        if not b or not t:
            return None, None, "缺数据"
        ds_list.append(t["dice"] - b["dice"])
        ses.append(math.sqrt(b.get("dice_std", 0) ** 2 / N[ds] + t.get("dice_std", 0) ** 2 / N[ds]))
    mean = sum(ds_list) / len(ds_list)
    se = math.sqrt(sum(s ** 2 for s in ses)) / len(ses) if ses else 0.0
    return mean, se, "ok"


dC, seC, stC = delta_mean("fad50")        # C 单独
dAC, seAC, stAC = delta_mean("fadaug50")  # A + C

lines = ["[decide_stage4] 方案C跨域结果 vs base50："]
if dC is not None:
    lines.append("  C单独(fad50)   跨域ΔDice = %+.2f (SE %.2f, |Δ|/SE=%.2f)" % (dC, seC, abs(dC) / seC if seC else 0))
else:
    lines.append("  C单独(fad50)   %s" % stC)
if dAC is not None:
    lines.append("  A+C(fadaug50)  跨域ΔDice = %+.2f (SE %.2f, |Δ|/SE=%.2f)" % (dAC, seAC, abs(dAC) / seAC if seAC else 0))
else:
    lines.append("  A+C(fadaug50)  %s" % stAC)

# 默认：D 单独（最保守、最能隔离频域损失这条正交路线）
FA, AUG, REASON = 0, 0, "方案C跨域无正向收益，Stage4 用 D 单独(FFL)干净检验频域监督信号"

cands = []
if dC is not None and dC > 0:
    cands.append((dC, 1, 0, "C单独为正，Stage4 = D+C 协同"))
if dAC is not None and dAC > 0:
    cands.append((dAC, 1, 1, "A+C为正，Stage4 = D+A+C 协同"))
if cands:
    best = max(cands, key=lambda x: x[0])
    _, FA, AUG, REASON = best

lines.append("[decide_stage4] 决策：freq_adapter=%d freq_aug=%d —— %s" % (FA, AUG, REASON))
report = "\n".join(lines)
print(report)

with open(OUT, "w") as f:
    f.write("# 由 decide_stage4.py 自动生成，供 run_night_stage3.sh source\n")
    f.write("FA=%d\n" % FA)
    f.write("AUG=%d\n" % AUG)
    f.write('REASON="%s"\n' % REASON.replace('"', "'"))
print("wrote:", OUT)
