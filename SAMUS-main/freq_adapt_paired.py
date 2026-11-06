#!/usr/bin/env python3
# =============================================================================
# freq_adapt_paired.py — 方案E 自适应γ / head×E 可加性 跨行配对分析 (Phase1 §36-§38)
#   背景：rag_segment 单次进程内的 paired_report 只能比 (同图 base vs method)。
#   但「自适应γ vs 固定γ vs base」「head×E vs head vs E」是跨配置对比，需在
#   同一确定性 loader(同图、同顺序)下按 image_name 匹配，做配对 Wilcoxon。
#   本脚本读 run_freq_adapt.sh 产出的研究日志(每行含 per.names / per.rag_dice / per.rag_se)，
#   对每个数据集输出下列判定，并给跨域均值与 §37a/§38 结论。
# 用法：python3 freq_adapt_paired.py freq_adapt_results.jsonl
# =============================================================================
import sys, json
import numpy as np
from scipy.stats import wilcoxon

XDOMAIN = ["DDTIz", "PTz"]      # 跨域均值只看这两集(§38 阈值 63.6 为跨域口径)


def load(path):
    """-> {(ds,tag): {name: {'dice':x,'se':y}}}；同名多次出现取最后一次。"""
    m = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        tag, ds = r.get("tag"), r.get("ds")
        per = r.get("per", {})
        names = per.get("names")
        if tag is None or ds is None or not names:
            continue
        d = {}
        rd = per.get("rag_dice")
        rs = per.get("rag_se")
        for i, nm in enumerate(names):
            d[str(nm)] = {"dice": float(rd[i]), "se": float(rs[i]) if rs else float("nan")}
        m[(ds, tag)] = d
    return m


def paired(a_map, b_map, key):
    """按共同 image_name 配对，返回 (n, meanΔ(B-A), wilcoxon p, per-image diff 数组)。"""
    common = [nm for nm in a_map if nm in b_map]
    if not common:
        return 0, float("nan"), float("nan"), np.array([])
    a = np.array([a_map[nm][key] for nm in common], dtype=float)
    b = np.array([b_map[nm][key] for nm in common], dtype=float)
    if np.isnan(a).any() or np.isnan(b).any():
        return len(common), float("nan"), float("nan"), np.array([])
    md = float((b - a).mean())
    try:
        _, p = wilcoxon(b, a, zero_method="wilcox")
    except Exception:
        p = float("nan")
    return len(common), md, float(p), b - a


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def mean_dice(m, ds, tag, key="dice"):
    d = m.get((ds, tag))
    if not d:
        return float("nan")
    v = [x[key] for x in d.values() if not np.isnan(x[key])]
    return float(np.mean(v)) if v else float("nan")


def report(m, ds):
    print(f"\n===== {ds} =====")
    tags = ["base", "efix", "e_adapt", "head", "raghead_e", "head_eadapt"]
    print("  %-11s %8s %8s" % ("tag", "meanDice", "meanSE"))
    for t in tags:
        print("  %-11s %8.2f %8.2f" % (t, mean_dice(m, ds, t, "dice"), mean_dice(m, ds, t, "se")))

    def line(ka, kb, key="dice"):
        n, md, p, _ = paired(m.get((ds, ka), {}), m.get((ds, kb), {}), key)
        print(f"  [{ka:>9} -> {kb:<9}] n={n:>3}  Δ{key}={md:+.2f} {star(p) if p==p else ''}(p={p:.3g})")
        return md, p

    print("  -- 自适应γ 主线(§37a) --")
    eadapt_vs_base = line("base", "e_adapt", "dice")
    line("base", "efix", "dice")
    line("efix", "e_adapt", "dice")          # 自适应 ≥ 固定?
    print("  -- head×E 可加性(§38) --")
    dE, _ = line("base", "raghead_e", "dice")       # head+E 总涨点
    dh, _ = line("base", "head", "dice")            # head 单独涨点
    de, _ = line("base", "efix", "dice")            # E 单独涨点
    line("head", "raghead_e", "dice")               # E 叠加到 head 的增量
    line("head_eadapt", "raghead_e", "dice")        # head×自适应 vs head×固定
    add = dh + de
    print(f"  >> 可加性: Δhead×E={dE:+.2f}  vs  Δhead({dh:+.2f})+ΔE({de:+.2f})={add:+.2f}"
          f"  交互项={dE-add:+.2f}")
    return dE, dh, de, add, eadapt_vs_base


def main():
    argv = [a for a in sys.argv[1:]]
    pair_tags = None
    if "--pair" in argv:
        i = argv.index("--pair")
        pair_tags = (argv[i + 1], argv[i + 2])
        argv = argv[:i] + argv[i + 3:]
    path = argv[0] if argv else "freq_adapt_results.jsonl"
    m = load(path)
    have = sorted({ds for (ds, _) in m})
    print(f"[paired] loaded {len(m)} (ds,tag) groups; datasets={have}")

    if pair_tags:                      # §52 通用两配置同图配对(A=基线, B=方法)
        ta, tb = pair_tags
        print(f"[paired] generic mode: {ta} -> {tb} (same-image Wilcoxon)")
        for ds in have:
            a, b = m.get((ds, ta)), m.get((ds, tb))
            if not a or not b:
                print(f"  {ds:<7} 缺 {ta} 或 {tb} 记录，跳过")
                continue
            n, md, p, _ = paired(a, b, "dice")
            _, mds, ps, _ = paired(a, b, "se")
            print(f"  {ds:<7} n={n:>3}  Δdice={md:+.2f} {star(p) if p==p else ''}(p={p:.3g})"
                  f"   Δse={mds:+.2f} {star(ps) if ps==ps else ''}"
                  f"   (SPC 逐图不可得, 见总表均值)")
        xdom = [ds for ds in XDOMAIN if (ds, ta) in m and (ds, tb) in m]
        if xdom:
            ma = np.mean([mean_dice(m, ds, ta) for ds in xdom])
            mb = np.mean([mean_dice(m, ds, tb) for ds in xdom])
            print(f"  跨域均值({'/'.join(xdom)}): {ta}={ma:.2f}  {tb}={mb:.2f}  Δ={mb-ma:+.2f}")
        return

    verdicts = {}
    for ds in have:
        verdicts[ds] = report(m, ds)

    # 跨域均值判定 (§38 阈值 63.6)
    xdom = [ds for ds in XDOMAIN if ds in m]
    if xdom:
        def xd(tag):
            vals = [mean_dice(m, ds, tag) for ds in xdom]
            vals = [v for v in vals if v == v]
            return float(np.mean(vals)) if vals else float("nan")
        print("\n===== 跨域均值(仅 %s) =====" % "/".join(xdom))
        for t in ["base", "efix", "e_adapt", "head", "raghead_e"]:
            print("  %-11s 跨域Dice=%.2f" % (t, xd(t)))
        print(f"\n  [§37a] 自适应γ 免训练 跨域={xd('e_adapt'):.2f}  vs 固定γ={xd('efix'):.2f}"
              f"  vs base={xd('base'):.2f}")
        print(f"  [§38]  head×E 跨域={xd('raghead_e'):.2f}  vs 63.6 门槛"
              f"  -> {'达/超(考虑 Phase2 重训)' if xd('raghead_e') > 63.6 else '未超(记非加性/跳过 2B)'}")


if __name__ == "__main__":
    main()
