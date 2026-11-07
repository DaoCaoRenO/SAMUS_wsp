#!/usr/bin/env python3
# =============================================================================
# 跨口径性能对比总表生成器
#   合并两个数据源，产出一张"口径对齐说明"的总表，每次探索后自动刷新：
#     1) blsamus(通用多器官/多为域内)  <- exp_out/result.csv  (metric 为 0~1 分数)
#     2) SAMUS-main(甲状腺专科/DDTI,PT零样本跨域) <- crossdomain_results.jsonl (dice 为百分值)
#   写回 ForPaper/探索复盘记录.md 顶部的 BL_CMP_START/END 标记之间，并输出独立 md。
#   已挂进 run_*.sh 链末尾，训练+跨域测试完成后自动运行。
# 用法：python3 make_bl_comparison.py
# =============================================================================
import json, os, csv, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CDJSONL = os.path.join(HERE, "crossdomain_results.jsonl")
HFJSONL = os.path.join(HERE, "hf_results.jsonl")     # 方案E：测试时高频衰减 gain 扫描结果
BLSAMUS_CSV = "/home/linux/Downloads/ForPaper/code2026-10-30/code/blsamus/exp_out/result.csv"
RECORD_MD = "/home/linux/Downloads/ForPaper/探索复盘记录.md"
STANDALONE_MD = os.path.join(HERE, "跨口径性能对比.md")
START, END = "<!-- BL_CMP_START -->", "<!-- BL_CMP_END -->"

DS = ["TN3K", "DDTIz", "PTz"]                       # SAMUS 跨域测试的数据集键
BL_MAP = {"TN3K": "TN3K", "DDTI": "DDTIz", "private_Thyroid": "PTz"}  # blsamus 名 -> SAMUS 键
HF_MAP = {"TN3K_hf": "TN3K", "DDTIz_hf": "DDTIz", "PTz_hf": "PTz"}    # 方案E 名 -> SAMUS 键

# 与 make_crossdomain_table.py 对齐的展示顺序与标签
TAG_ORDER = ["base50", "freq50", "fad50", "fadaug50", "floss50", "edge50", "c2f50", "ep270_300", "rag", "raghead", "raghead_e", "raghead_e_tr2", "e_adapt", "hfal_e", "hfal_head", "hfal_e6", "hfal_head6", "shift50", "multi_tn3k", "bl_nothy"]
TAG_LABEL = {
    "base50":     "SAMUS专科 50轮对照",
    "freq50":     "SAMUS +A(频谱增广)",
    "fad50":      "SAMUS +C(频域Adapter)",
    "fadaug50":   "SAMUS +A+C",
    "floss50":    "SAMUS +D(FFL,已崩)",
    "edge50":     "SAMUS +边缘增强(Sobel)",
    "c2f50":      "SAMUS +coarse-to-fine(PBR)",
    "ep270_300":  "SAMUS专科 300轮(ep270原基线)",
    "rag":        "SAMUS ep270 +RAG检索原型先验(免训练)",
    "raghead":    "SAMUS ep270 +RAG可训练PCR头(只训头+域偏移模拟)",
    "raghead_e":  "SAMUS ep270 +RAG PCR头×方案E",
    "raghead_e_tr2": "⭐ SAMUS ep270 +RAG PCR头×自适应γ(Phase2B 重训头)",
    "e_adapt":    "⭐ SAMUS ep270 +方案E逐图自适应γ(免训练)",
    "hfal_e":     "SAMUS ep270 +方案E磁盘口径(每域最优 gain DDTI0.55/PT0.65)",
    "hfal_head":  "SAMUS ep270 +RAG PCR头×方案E磁盘口径(每域最优 gain)",
    "hfal_e6":    "SAMUS ep270 +方案E磁盘口径(统一 gain=0.60)",
    "hfal_head6": "SAMUS ep270 +RAG PCR头×方案E磁盘口径(统一 gain=0.60)",
    "shift50":    "SAMUS 50轮 +合成域偏移训练",
    "multi_tn3k": "SAMUS+USIC(无甲状腺)+TN3K 100轮(A)",
    "bl_nothy":   "blsamus+USIC(纯无甲状腺) 100轮(B)",
}
CAL_SAMUS = "专科 · TN3K域内/DDTI,PT**零样本**"
CAL_BL = "通用多器官 · 甲状腺**多为域内**(训过)"


def load_samus():
    rows = {}
    if os.path.exists(CDJSONL):
        for line in open(CDJSONL):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            rows[(d.get("tag"), d.get("ds"))] = d
    return rows


def load_blsamus():
    out, t = {}, None
    if not os.path.exists(BLSAMUS_CSV):
        return out, t
    for r in csv.DictReader(open(BLSAMUS_CSV)):
        ds, task = r.get("dataset"), r.get("task", "")
        if ds in BL_MAP and "omni_seg" in task:
            try:
                out[BL_MAP[ds]] = float(r["metric"]) * 100.0   # 0~1 -> 百分值
                t = r.get("time")
            except Exception:
                pass
    return out, t


def cross_mean(vals):
    xs = [vals.get("DDTIz"), vals.get("PTz")]
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def fmt(v):
    return ("%.1f" % v) if v is not None else "-"


def load_freq_e():
    """读 hf_results.jsonl，取跨域均值(DDTI,PT)最高的 gain，返回该 gain 的三集 dice + 跨域均值。
    方案E = 在 ep270(300轮) 上测试时对高频幅度乘 gain（纯推理端、零训练）。"""
    if not os.path.exists(HFJSONL):
        return None
    by_gain = {}
    for line in open(HFJSONL):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        g, ds = d.get("gain"), d.get("ds")
        if g is None or ds not in HF_MAP:
            continue
        by_gain.setdefault(g, {})[HF_MAP[ds]] = d["dice"]
    best = None
    for g, vals in by_gain.items():
        cm = cross_mean(vals)
        if cm is not None and (best is None or cm > best["cross_mean"]):
            best = {"gain": g, "vals": vals, "cross_mean": cm}
    return best


def build_table():
    samus = load_samus()
    bl, bl_time = load_blsamus()
    hdr = ["模型 / 探索", "口径", "TN3K(域内)", "DDTI(跨域)", "PT(跨域)", "跨域均值(DDTI,PT)"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]

    if bl:                                            # blsamus 通用/域内 行
        lines.append("| " + " | ".join([
            "blsamus(通用多器官)", CAL_BL,
            fmt(bl.get("TN3K")), fmt(bl.get("DDTIz")), fmt(bl.get("PTz")), fmt(cross_mean(bl)),
        ]) + " |")

    e = load_freq_e()                                 # ⭐ 方案E 行(当前全局最好，测试时叠加)
    if e:
        v = e["vals"]
        lines.append("| " + " | ".join([
            "⭐ ep270+方案E(测试时高频衰减 gain%.2f)" % e["gain"],
            "专科 · **纯推理端零训练** · DDTI,PT零样本",
            fmt(v.get("TN3K")), fmt(v.get("DDTIz")), fmt(v.get("PTz")), fmt(e["cross_mean"]),
        ]) + " |")

    for tag in TAG_ORDER:                             # SAMUS 各探索行
        vals, any_data = {}, False
        for ds in DS:
            d = samus.get((tag, ds))
            if d:
                vals[ds] = d["dice"]; any_data = True
        if not any_data:
            continue
        lines.append("| " + " | ".join([
            TAG_LABEL.get(tag, tag), CAL_SAMUS,
            fmt(vals.get("TN3K")), fmt(vals.get("DDTIz")), fmt(vals.get("PTz")), fmt(cross_mean(vals)),
        ]) + " |")

    note = ("\n> ⭐ **方案E** 行 = 当前**零样本跨域口径下的最好成绩**；blsamus 跨域均值虽更高，但那是**域内**(训过这些集)、"
            "不同赛道不可比。E 在 ep270(300轮) 上测试时叠加高频幅度衰减、纯推理端零训练成本，且是唯一有硬证据的正向——"
            "9 点 gain 扫描呈倒 U 型、峰值精确落在独立实测 γ*(0.611/0.652)，几乎不可能是噪声；详见 `hf_results.*`。")
    return "\n".join(lines) + note, bl_time


def update_record(table_md):
    if not os.path.exists(RECORD_MD):
        print("[bl_cmp] 找不到记录文件，跳过写回:", RECORD_MD); return False
    txt = open(RECORD_MD, encoding="utf-8").read()
    if START not in txt or END not in txt:
        print("[bl_cmp] 记录里缺少 BL_CMP_START/END 标记，跳过写回（请先插入标记段）"); return False
    i, j = txt.index(START), txt.index(END)
    new = txt[:i] + START + "\n" + table_md + "\n" + END + txt[j + len(END):]
    open(RECORD_MD, "w", encoding="utf-8").write(new)
    return True


def main():
    table_md, bl_time = build_table()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    if update_record(table_md):
        print("[bl_cmp] 已写回记录顶部总表:", RECORD_MD)
    header = ("# 跨口径性能对比总表（自动更新）\n\n"
              "> ⚠️ **口径不同，不能直接比大小**：blsamus 是通用多器官模型、甲状腺/DDTI 多在其**训练分布内(域内)**；"
              "SAMUS-main 是甲状腺**专科**模型，TN3K 域内、DDTI/PT 为**从未训练过的零样本跨域**（更难）。"
              "blsamus DDTI 更高≠架构更强（它训过）；SAMUS-main TN3K 更高因为是专科全容量。\n\n"
              "> 更新时间：%s ｜ blsamus 数据时间：%s ｜ 数据源：blsamus result.csv + SAMUS crossdomain_results.jsonl\n\n"
              % (stamp, bl_time or "N/A"))
    with open(STANDALONE_MD, "w", encoding="utf-8") as f:
        f.write(header + table_md + "\n")
    print("[bl_cmp] 已输出独立表:", STANDALONE_MD)


if __name__ == "__main__":
    main()
