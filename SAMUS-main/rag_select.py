#!/usr/bin/env python3
# =============================================================================
# rag_select.py — 在 TN3K `val`(499) 上网格选参（k×λ×τ），锁参后一次性报跨域。
#   防泄漏纪律（计划 §4）：只在源域 val 选，绝不在 DDTIz/PTz 测试集调 λ。
#   工程：模型+记忆库+val 特征只算一次并缓存(imge/base_logits/e_q/gt)，网格内仅
#         做检索→聚合→先验→融合→指标（纯张量+每图 dice/se），大幅省时。
#   选参目标：先最大化 val SEN(召回，计划核心假设)，Dice 作平手判据。
#   输出：rag_valbest.json（锁定的 k/λ/τ + val 详情），供 run_rag.sh 读取。
# =============================================================================
import os, json, argparse, time, itertools
import numpy as np
import torch
import torch.nn.functional as F
import rag_common as R
from rag_segment import aggregate, hfa_boost

KS = [1, 5, 10]
LAMS = [0.25, 0.5, 1.0]
TAUS = [4.0, 8.0, 16.0]


@torch.no_grad()
def cache_queries(model, opt, args):
    split = "val"
    loader = R.make_loader(opt, split, batch_size=args.batch)
    store = {"imge": [], "base": [], "eq": [], "gt": []}
    t0 = time.time()
    for bi, dp in enumerate(loader):
        imgs = dp["image"].to(opt.device, dtype=torch.float32)
        if args.stack_e != 1.0:
            imgs = hfa_boost(imgs, gain=args.stack_e)
        imge = R.dense_feat(model, imgs)
        base = R.model_base_logits(model, imgs)
        locs = imge.flatten(2).transpose(1, 2)
        eq = F.normalize(locs.mean(1), dim=1)
        gt = (dp["label"].to(opt.device, dtype=torch.float32)[:, 0] == 1)
        store["imge"].append(imge.float().cpu())
        store["base"].append(base.float().cpu())
        store["eq"].append(eq.float().cpu())
        store["gt"].append(gt.cpu())
        if bi % 20 == 0:
            print(f"  cache {bi}/{len(loader)} {time.time()-t0:.1f}s", flush=True)
    return {k: torch.cat(v) for k, v in store.items()}


def evaluate(C, E_retr, bank_gpu, k, lam, tau, device, rtemp=0.1):
    N = C["imge"].shape[0]
    imge = C["imge"].to(device); base = C["base"].to(device); eq = C["eq"].to(device)
    idx, sim = R.retrieve(eq, E_retr, k)
    w = F.softmax(sim / rtemp, dim=1)
    Pfg, Pbg = aggregate(bank_gpu, idx, w, "softmax")
    prior = R.similarity_map(imge, Pfg, Pbg, tau)
    fused = R.fuse(base, prior, lam)
    seg = (torch.sigmoid(fused)[:, 0] > 0.5).cpu().numpy()
    gt = C["gt"].numpy()
    dices, ses = [], []
    for j in range(N):
        m = R.seg_metrics(seg[j], gt[j])
        dices.append(m[0]); ses.append(m[3])
    return float(np.mean(dices)), float(np.mean(ses))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth")
    ap.add_argument("--bank", default=os.path.join(R.HERE, "rag_bank", "tn3k_bank.pt"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--retr", choices=["roi", "mean"], default="roi")
    ap.add_argument("--stack-e", dest="stack_e", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(R.HERE, "rag_valbest.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, opt, ns = R.build_model(task="TN3K", load_path=args.ckpt)
    bank = torch.load(args.bank, map_location="cpu")
    E_retr = bank["E_roi" if args.retr == "roi" else "E_mean"].to(device)
    bank_gpu = {"E_roi": bank["E_roi"].to(device), "P_bg": bank["P_bg"].to(device)}

    C = cache_queries(model, opt, args)
    # λ=0 基线（对照组）
    b_dice, b_se = evaluate(C, E_retr, bank_gpu, KS[0], 0.0, 0.0, device)
    print(f"[select] val base(λ=0): dice={b_dice:.2f} se={b_se:.2f}", flush=True)

    results = []
    for k, lam, tau in itertools.product(KS, LAMS, TAUS):
        d, s = evaluate(C, E_retr, bank_gpu, k, lam, tau, device)
        results.append({"k": k, "lambda": lam, "tau": tau, "val_dice": d, "val_se": s})
        print(f"  k={k:>2} λ={lam:<4} τ={tau:<4}  val_dice={d:.2f} val_se={s:.2f}", flush=True)
    # 选参（对齐计划成功判据 §54）：在「val Dice 不回退于 base」的可行域内最大化 SEN；
    # 若可行域为空则退化为纯 SEN 最大。Dice 作平手判据。这样锁定的是 Pareto 提升点，
    # 而非牺牲 Dice 换 SEN 的极端点。
    feasible = [r for r in results if r["val_dice"] >= b_dice - 1e-9]
    pool = feasible if feasible else results
    best = max(pool, key=lambda r: (round(r["val_se"], 3), r["val_dice"]))
    payload = {"retr": args.retr, "stack_e": args.stack_e,
               "base": {"val_dice": b_dice, "val_se": b_se},
               "best": best, "grid": results}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[select] BEST k={best['k']} λ={best['lambda']} τ={best['tau']} "
          f"val_dice={best['val_dice']:.2f} val_se={best['val_se']:.2f} -> {args.out}")


if __name__ == "__main__":
    main()
