#!/usr/bin/env python3
# 读取 crossdomain_results.jsonl，汇总成 CSV + Markdown 对比表
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, "crossdomain_results.jsonl")
CSV = os.path.join(HERE, "crossdomain_results.csv")
MD = os.path.join(HERE, "crossdomain_results.md")

DS_ORDER = ["TN3K", "DDTIz", "PTz"]           # TN3K=域内, 其余=跨域零样本
DS_LABEL = {"TN3K": "TN3K(域内)", "DDTIz": "DDTI(跨域)", "PTz": "private_Thyroid(跨域)"}
TAG_ORDER = ["base50", "freq50", "fad50", "fadaug50", "floss50", "edge50", "c2f50", "ep270_300", "rag", "raghead", "raghead_e", "e_adapt", "hfal_e", "hfal_head", "hfal_e6", "hfal_head6", "shift50", "multi_tn3k", "bl_nothy"]
TAG_LABEL = {
    "base50":     "SAMUS 50轮 对照(无A无C)",
    "freq50":     "SAMUS 50轮 +A(频谱随机化增广)",
    "fad50":      "SAMUS 50轮 +C(Block内频域Adapter)",
    "fadaug50":   "SAMUS 50轮 +A+C",
    "floss50":    "SAMUS 50轮 +D(FFL频域损失,已崩)",
    "edge50":     "SAMUS 50轮 +边缘增强(Sobel边界损失)",
    "c2f50":      "SAMUS 50轮 +coarse-to-fine(PBR渐进边界精炼)",
    "ep270_300":  "SAMUS 300轮(原基线 ep270)",
    "rag":        "ep270 +RAG原型检索先验融合(免训练)",
    "raghead":    "ep270 +RAG可训练PCR头(原型条件化精炼,只训头+域偏移模拟)",
    "raghead_e":  "ep270 +RAG PCR头×方案E(高频衰减叠加可学习头)",
    "e_adapt":    "⭐ ep270 +方案E逐图自适应γ(免训练,源域高频能量自校准)",
    "hfal_e":     "ep270 +方案E磁盘口径增益(每域实测最优 gain DDTI0.55/PT0.65;≈63.6 基线)",
    "hfal_head":  "ep270 +RAG PCR头×方案E磁盘口径(每域最优 gain;与 63.6 同尺的 head×E)",
    "hfal_e6":    "ep270 +方案E磁盘口径增益(三集统一 gain=0.60)",
    "hfal_head6": "ep270 +RAG PCR头×方案E磁盘口径(三集统一 gain=0.60)",
    "shift50":    "SAMUS 50轮 +合成域偏移训练(全量train_shift对症对照)",
    "multi_tn3k": "SAMUS+USIC(无甲状腺)+TN3K 100轮(A)",
    "bl_nothy":   "blsamus+USIC(纯无甲状腺) 100轮(B)",
}

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

def _f(v, dash="-"):
    """%.2f 或 None→dash (B 的 bl_nothy 行仅有 dice, 其余指标 N/A)。"""
    return dash if v is None else ("%.2f" % v)

# ---------------- CSV（长表：每行一个 model×dataset） ----------------
with open(CSV, "w") as f:
    f.write("model,dataset,dice,dice_std,iou,acc,sen,spec,hd95_manhattan\n")
    for tag in TAG_ORDER:
        for ds in DS_ORDER:
            d = rows.get((tag, ds))
            if not d:
                continue
            f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                tag, ds, _f(d["dice"]), _f(d.get("dice_std", 0.0)), _f(d["iou"]),
                _f(d["acc"]), _f(d["se"]), _f(d["sp"]), _f(d["hdis"])))

# ---------------- Markdown（以 Dice 为主表 + 全指标明细） ----------------
lines = ["# SAMUS 频域改造 A/C 因子消融 — 跨域零样本对比\n"]
lines.append("A = 训练时频谱随机化增广；C = ParaBlock 内并联可学习复数滤波 Freq_Adapter。"
             "四组均为 50 轮、同 seed/同初始化/lr 1e-4/bs 8，唯一变量为 A/C 开关。\n")
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
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            TAG_LABEL.get(tag, tag), ds, _f(d["dice"], "N/A"), _f(d["iou"], "N/A"),
            _f(d["acc"], "N/A"), _f(d["se"], "N/A"), _f(d["sp"], "N/A"), _f(d["hdis"], "N/A")))

# ---------------- 2×2 因子消融（均相对 base50 的 ΔDice） ----------------
lines.append("\n**2×2 因子消融** ΔDice（相对 50轮对照组 base50，百分点；正=有收益）:\n")
fhdr = ["数据集", "无A无C", "+A", "+C", "+A+C"]
lines.append("| " + " | ".join(fhdr) + " |")
lines.append("|" + "|".join(["---"] * len(fhdr)) + "|")
for ds in DS_ORDER:
    b = rows.get(("base50", ds))
    cells = [DS_LABEL[ds], "0.00 (基准)"]
    for tag in ("freq50", "fad50", "fadaug50"):
        d = rows.get((tag, ds))
        cells.append("%+.2f" % (d["dice"] - b["dice"]) if (b and d) else "-")
    lines.append("| " + " | ".join(cells) + " |")

# 与 300 轮原基线的差距（50 轮预算下的代价）
lines.append("\n与 300轮原基线(ep270) 的 ΔDice（50轮各组 − ep270，负=轮数不足代价）:\n")
lines.append("| 数据集 | base50 | +A | +C | +A+C |")
lines.append("|---|---|---|---|---|")
for ds in DS_ORDER:
    e = rows.get(("ep270_300", ds))
    cells = [DS_LABEL[ds]]
    for tag in ("base50", "freq50", "fad50", "fadaug50"):
        d = rows.get((tag, ds))
        cells.append("%+.2f" % (d["dice"] - e["dice"]) if (e and d) else "-")
    lines.append("| " + " | ".join(cells) + " |")

with open(MD, "w") as f:
    f.write("\n".join(lines) + "\n")

print("wrote:", CSV)
print("wrote:", MD)
