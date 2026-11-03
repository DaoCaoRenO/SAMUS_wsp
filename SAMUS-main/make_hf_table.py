#!/usr/bin/env python3
"""汇总方案E（测试时高频幅度增益扫描）的 hf_results.jsonl -> CSV + Markdown"""
import json, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, "hf_results.jsonl")
CSV = os.path.join(HERE, "hf_results.csv")
MD = os.path.join(HERE, "hf_results.md")
N = {"TN3K_hf": 379, "DDTIz_hf": 47, "PTz_hf": 44}
DS_ORDER = ["TN3K_hf", "DDTIz_hf", "PTz_hf"]
DS_LABEL = {"TN3K_hf": "TN3K(域内)", "DDTIz_hf": "DDTI(跨域)", "PTz_hf": "private_Thyroid(跨域)"}
# 无增益的原始基线（ep270，来自 crossdomain_results.jsonl）
ORIG = {"TN3K_hf": 80.99, "DDTIz_hf": 58.32, "PTz_hf": 62.15}

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
        rows[(round(float(d["gain"]), 3), d["ds"])] = d

gains = sorted({g for g, _ in rows})

with open(CSV, "w") as f:
    f.write("gain,dataset,dice,dice_std,iou,acc,sen,spec,hd95_manhattan\n")
    for g in gains:
        for ds in DS_ORDER:
            d = rows.get((g, ds))
            if d:
                f.write("%.3f,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (
                    g, ds, d["dice"], d.get("dice_std", 0), d["iou"], d["acc"], d["se"], d["sp"], d["hdis"]))

L = ["# 方案E：测试时高频幅度增益扫描（zero-training）\n"]
L.append("模型统一为 SAMUS ep270(300轮)。对 r>=0.25 的高频幅度乘以 gain，相位不变、GT 不变；"
         "gain=1.00 为对照组（仅多一次 resize+JPEG 重编码，用于校验链路可信度）。\n")
meas = os.path.join(HERE, "hf_measure.txt")
if os.path.exists(meas):
    L.append("实测高频失配（决定扫描方向的依据）:\n")
    L.append("```")
    L.append(open(meas).read().strip())
    L.append("```\n")

hdr = ["gain"] + [DS_LABEL[d] for d in DS_ORDER]
L.append("| " + " | ".join(hdr) + " |")
L.append("|" + "|".join(["---"] * len(hdr)) + "|")
for g in gains:
    cells = ["%.2f" % g + (" (对照)" if abs(g - 1.0) < 1e-9 else "")]
    for ds in DS_ORDER:
        d = rows.get((g, ds))
        cells.append("%.2f" % d["dice"] if d else "-")
    L.append("| " + " | ".join(cells) + " |")

L.append("\n相对**原始未处理基线**的 ΔDice（正=涨点）:\n")
L.append("| gain | " + " | ".join(DS_LABEL[d] for d in DS_ORDER) + " | 跨域均值Δ |")
L.append("|---|---|---|---|")
for g in gains:
    cells, cross = [], []
    for ds in DS_ORDER:
        d = rows.get((g, ds))
        if d:
            delta = d["dice"] - ORIG[ds]
            cells.append("%+.2f" % delta)
            if ds != "TN3K_hf":
                cross.append(delta)
        else:
            cells.append("-")
    cells.append("%+.2f" % (sum(cross) / len(cross)) if cross else "-")
    L.append("| %.2f | %s |" % (g, " | ".join(cells)))

# 显著性：对照组 gain=1.0 与各 gain 的差，附 SEM 判据
L.append("\n显著性判据（|Δ|/SE ≥ 2 才算真信号；SE 由逐图 std/√n 合成）:\n")
L.append("| gain | 数据集 | Δ vs 对照(1.00) | 合并SE | \\|Δ\\|/SE | 判定 |")
L.append("|---|---|---|---|---|---|")
for g in gains:
    for ds in DS_ORDER:
        c = rows.get((1.0, ds)); d = rows.get((g, ds))
        if not c or not d or abs(g - 1.0) < 1e-9:
            continue
        se = math.sqrt(c["dice_std"] ** 2 / N[ds] + d["dice_std"] ** 2 / N[ds])
        delta = d["dice"] - c["dice"]
        ratio = abs(delta) / se if se > 0 else 0
        L.append("| %.2f | %s | %+.2f | %.2f | %.2f | %s |" % (
            g, DS_LABEL[ds], delta, se, ratio, "显著" if ratio >= 2 else "噪声内"))

with open(MD, "w") as f:
    f.write("\n".join(L) + "\n")
print("wrote:", CSV)
print("wrote:", MD)
