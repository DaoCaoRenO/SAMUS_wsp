#!/usr/bin/env python3
# 读取 crossdomain_results.jsonl，汇总成 CSV + Markdown 对比表
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, "crossdomain_results.jsonl")
CSV = os.path.join(HERE, "crossdomain_results.csv")
MD = os.path.join(HERE, "crossdomain_results.md")

DS_ORDER = ["TN3K", "DDTIz", "PTz"]           # TN3K=域内, 其余=跨域零样本
DS_LABEL = {"TN3K": "TN3K(域内)", "DDTIz": "DDTI(跨域)", "PTz": "private_Thyroid(跨域)"}
TAG_ORDER = ["base50", "freq50", "ep270_300"]
TAG_LABEL = {"base50": "SAMUS 50轮(基线,无频域)", "freq50": "SAMUS 50轮(频谱增广)", "ep270_300": "SAMUS 300轮(原基线 ep270)"}

rows = {}
if os.path.exists(JSONL):
    for line in open(JSONL):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        rows[(d.get("tag", "?"), d.get("ds", "?"))] = d

# ---------------- CSV（长表：每行一个 model×dataset） ----------------
with open(CSV, "w") as f:
    f.write("model,dataset,dice,dice_std,iou,acc,sen,spec,hd95_manhattan\n")
    for tag in TAG_ORDER:
        for ds in DS_ORDER:
            d = rows.get((tag, ds))
            if not d:
                continue
            f.write("%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (
                tag, ds, d["dice"], d.get("dice_std", 0.0), d["iou"],
                d["acc"], d["se"], d["sp"], d["hdis"]))

# ---------------- Markdown（以 Dice 为主表 + 全指标明细） ----------------
lines = ["# SAMUS 频谱增广 A/B — 跨域零样本对比\n"]
lines.append("主指标 **Dice(%)**（越大越好；DDTI/PT 为跨库零样本，HD95 为曼哈顿距离仅供参考）:\n")
hdr = ["模型"] + [DS_LABEL[ds] for ds in DS_ORDER]
lines.append("| " + " | ".join(hdr) + " |")
lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
for tag in TAG_ORDER:
    cells = [TAG_LABEL.get(tag, tag)]
    any_data = False
    for ds in DS_ORDER:
        d = rows.get((tag, ds))
        if d:
            cells.append("%.2f" % d["dice"]); any_data = True
        else:
            cells.append("-")
    if any_data:
        lines.append("| " + " | ".join(cells) + " |")

lines.append("\n全指标明细:\n")
mhdr = ["模型", "数据集", "Dice", "IoU", "ACC", "SEN", "SPC", "HD95"]
lines.append("| " + " | ".join(mhdr) + " |")
lines.append("|" + "|".join(["---"] * len(mhdr)) + "|")
for tag in TAG_ORDER:
    for ds in DS_ORDER:
        d = rows.get((tag, ds))
        if not d:
            continue
        lines.append("| %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
            TAG_LABEL.get(tag, tag), ds, d["dice"], d["iou"], d["acc"], d["se"], d["sp"], d["hdis"]))

# 频谱增广净增益（freq50 - base50），仅跨域两列
lines.append("\n频谱增广净增益 ΔDice（freq50 − base50，百分点）:\n")
lines.append("| 数据集 | ΔDice |")
lines.append("|---|---|")
for ds in DS_ORDER:
    b = rows.get(("base50", ds)); fr = rows.get(("freq50", ds))
    if b and fr:
        lines.append("| %s | %+.2f |" % (ds, fr["dice"] - b["dice"]))

with open(MD, "w") as f:
    f.write("\n".join(lines) + "\n")

print("wrote:", CSV)
print("wrote:", MD)
