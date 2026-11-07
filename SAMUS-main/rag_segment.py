#!/usr/bin/env python3
# =============================================================================
# rag_segment.py — query 时：检索→聚合原型→相似度先验图→融合→评测（免训练）
#   对应计划 §3。严格复用 rag_common（=test.py/eval_mask_slice2 同口径），保证可与
#   crossdomain 总表直接对比。λ=0 即 SAMUS-only 对照组（同图两值 → 支持配对检验）。
#
#   流程（每张 query，无 GT 参与检索/选参）：
#     1) imge_q=image_encoder(imgs)；e_q=均值池化 L2norm（非对称 query 侧）
#     2) 暴力余弦在 TN3K train 记忆库取 top-k（--retr roi|mean 切换检索向量=消融）
#     3) 相似度 softmax 权重 w_i（--agg softmax|top1 加权聚合 vs 固定最近=消融）
#        聚合前景原型 P_fg=Σ w_i p_fg^i、背景 P_bg=Σ w_i p_bg^i，各自重归一
#     4) 相似度先验图 S=τ·(cos(loc,P_fg)-cos(loc,P_bg)) (32×32) → 双线性上采样
#     5) fused = base_logits + λ·prior → sigmoid>0.5 出 mask
#     6) --stack_e：先对 query 图做测试时高频增益(方案E)再走上面(验证与 RAG 是否正交互补)
#   结果写 rag_results.jsonl（含每图 base/rag 的 dice、se，供配对检验）。
# =============================================================================
import os, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import rag_common as R


def hfa_boost(img, gain=1.5, ratio=0.25):
    """(已上提至 rag_common，保留薄封装向后兼容)对 (B,3,H,W)∈[0,1] 高频幅度(r>=ratio)乘 gain。"""
    return R.hfa_boost(img, gain=gain, ratio=ratio)


def aggregate(bank, idx, w, agg):
    """按 top-k 权重聚合前景/背景原型。idx(B,k) w(B,k) → P_fg,P_bg(B,256) 已重归一。"""
    if agg == "top1":
        w = F.one_hot(idx[:, 0], num_classes=bank["E_roi"].shape[0]).float()  # (B,N) one-hot
        pf = w @ bank["E_roi"]; pb = w @ bank["P_bg"]
    else:
        pf = torch.bmm(w.unsqueeze(1), bank["E_roi"][idx]).squeeze(1)         # (B,k)*(B,k,256)
        pb = torch.bmm(w.unsqueeze(1), bank["P_bg"][idx]).squeeze(1)
    return F.normalize(pf, dim=1), F.normalize(pb, dim=1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="DDTIz|PTz|TN3K")
    ap.add_argument("--split", default="test", help="val(选参) | test(评测)")
    ap.add_argument("--ckpt", default="checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth")
    ap.add_argument("--bank", default=os.path.join(R.HERE, "rag_bank", "tn3k_bank.pt"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=8.0)
    ap.add_argument("--retr", choices=["roi", "mean"], default="roi", help="检索向量: ROI加权|均值")
    ap.add_argument("--agg", choices=["softmax", "top1"], default="softmax", help="原型聚合方式")
    ap.add_argument("--retr-temp", dest="rtemp", type=float, default=0.1, help="softmax 温度(越小越锐)")
    ap.add_argument("--stack-e", dest="stack_e", type=float, default=1.0, help="方案E 高频增益(固定 γ；1.0=关)")
    ap.add_argument("--e-adapt", dest="e_adapt", type=float, default=None,
                    help="逐图自适应 γ(方案E 升级)，值为高频半径 ratio(如 0.25)；与 --stack-e 互斥")
    ap.add_argument("--e-ref", default=os.path.join(R.HERE, "rag_bank", "hf_ref.json"),
                    help="源域高频能量参考 json(缓存 E_src_ref)")
    ap.add_argument("--e-ref-refresh", dest="e_ref_refresh", action="store_true",
                    help="强制重算并回写 --e-ref")
    ap.add_argument("--e-lo", dest="e_lo", type=float, default=0.5, help="自适应 gain 下限 clamp")
    ap.add_argument("--e-hi", dest="e_hi", type=float, default=1.3, help="自适应 gain 上限 clamp")
    ap.add_argument("--src-task", dest="src_task", default="TN3K", help="自适应 γ 参考源域 task")
    ap.add_argument("--src-split", dest="src_split", default="train", help="自适应 γ 参考源域 split")
    ap.add_argument("--gate", choices=["off", "conf", "hole"], default="off",
                    help="先验注入方式：全局均匀 off | 置信度门控 conf | 候选框内部空洞填充 hole")
    ap.add_argument("--gscale", type=float, default=4.0, help="置信门控尺度 s（g=exp(-|logit|/s)）")
    ap.add_argument("--head_ckpt", default="", help="给定则用训练好的 PCR 头产 masks_fine(替代固定 λ 融合)")
    ap.add_argument("--head_hidden", type=int, default=32, help="PCR 头隐藏通道数(需与训练一致)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(R.HERE, "rag_results.jsonl"))
    ap.add_argument("--emit-table", dest="emit_table", default="",
                    help="给定路径时，把本次(锁定参数)结果 upsert 为该跨域总表 jsonl 的一行 rag 记录")
    ap.add_argument("--tag", default="", help="覆盖自动标签(rag/raghead); 用于纯基线评测如 multi_tn3k/bl_nothy")
    ap.add_argument("--ds-key", dest="ds_key", default="",
                    help="记录里 ds 字段的归组名(默认=args.task)。口径对齐实验用：把原图 base 与 "
                         "磁盘增强集 DDTIz_hf_g06 都归到 ds=DDTIz，freq_adapt_paired 才能按 image_name 跨行配对")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, opt, ns = R.build_model(task=args.task, load_path=args.ckpt)
    bank = torch.load(args.bank, map_location="cpu")
    E_retr = bank["E_roi" if args.retr == "roi" else "E_mean"].to(device)
    bank_gpu = {"E_roi": bank["E_roi"].to(device), "P_bg": bank["P_bg"].to(device)}

    # ---- 方案E 逐图自适应 γ：加载/计算源域高频能量参考 E_src_ref ----
    E_src_ref = None
    if args.e_adapt is not None:
        if args.stack_e != 1.0:
            raise SystemExit("[rag] --e-adapt 与 --stack-e 互斥(同为方案E，只是自适应/固定之别)")
        ratio = args.e_adapt
        ref = None
        if os.path.exists(args.e_ref) and not args.e_ref_refresh:
            try:
                j = json.load(open(args.e_ref))
                if abs(float(j.get("ratio", -1)) - ratio) < 1e-9 and "median" in j:
                    ref = j
            except Exception:
                ref = None
        if ref is None:
            ref = R.source_hf_ref(task=args.src_task, split=args.src_split, ratio=ratio, device=device)
            os.makedirs(os.path.dirname(args.e_ref), exist_ok=True)
            json.dump(ref, open(args.e_ref, "w"))
            print(f"[rag] computed E_src_ref -> {args.e_ref}: {ref}", flush=True)
        E_src_ref = float(ref["median"])
        print(f"[rag] E-adapt ON ratio={ratio} E_src_ref(median)={E_src_ref:.3f} "
              f"clamp=[{args.e_lo},{args.e_hi}] src={ref['src']}", flush=True)

    head_on = bool(args.head_ckpt)
    if head_on:
        from rag_head import build_pcr_head
        model.pcr = build_pcr_head(hidden=args.head_hidden).to(device)
        sd = torch.load(args.head_ckpt, map_location="cpu")
        sd = {(k[7:] if k[:7] == "module." else k): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_pcr = sum(1 for k in sd if k.startswith("pcr."))
        print(f"[rag] HEAD ON ckpt={args.head_ckpt} pcr_keys={n_pcr} "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        assert n_pcr > 0, "head_ckpt 里没有 pcr.* 键，不是 PCR 头训出来的"
        model.eval()

    split = opt.test_split if args.split == "test" else opt.val_split
    if args.task == "TN3K":
        split = "val" if args.split == "val" else opt.test_split
    loader = R.make_loader(opt, split, batch_size=args.batch)
    print(f"[rag] task={args.task} split={split} k={args.k} λ={args.lam} τ={args.tau} "
          f"retr={args.retr} agg={args.agg} stack_e={args.stack_e} gate={args.gate} gscale={args.gscale}", flush=True)

    names, base_arr, rag_arr = [], [], []
    t0 = time.time()
    for bi, dp in enumerate(loader):
        imgs = dp["image"].to(device, dtype=torch.float32)
        gt = (dp["label"].to(device, dtype=torch.float32)[:, 0] == 1)   # (B,256,256) bool
        if args.e_adapt is not None:
            e_i = R.hf_energy_tensor(imgs, args.e_adapt)                 # (B,) 每图高频能量
            gain = (E_src_ref / e_i.clamp_min(1e-6)).clamp(args.e_lo, args.e_hi)
            imgs = R.hfa_boost(imgs, gain=gain, ratio=args.e_adapt)
        elif args.stack_e != 1.0:
            imgs = hfa_boost(imgs, gain=args.stack_e)
        imge = R.dense_feat(model, imgs)                    # (B,256,32,32)
        base_logits = R.model_base_logits(model, imgs)      # (B,1,256,256)
        # leave-one-out: 传 batch 内图片名给 exclude(源域 bank 自然 no-op，目标域 bank 才屏蔽自身)
        batch_names = [dp["image_name"][j] if isinstance(dp["image_name"], (list, tuple))
                       else str(dp["image_name"][j]) for j in range(imgs.shape[0])]
        if head_on:
            # 可学习 PCR 头：检索原型逐位置余弦→残差精炼，替代固定 λ 手工融合
            sim_fg, sim_bg = R.prototypes_for_query(imge, bank, args.k, exclude=batch_names, retr=args.retr)
            fused = model.pcr(imge, base_logits, sim_fg, sim_bg)            # masks_fine (B,1,256,256)
        else:
            # query 全局检索向量(非对称: 均值池化)
            locs = imge.flatten(2).transpose(1, 2)              # (B,1024,256)
            e_q = F.normalize(locs.mean(1), dim=1)
            # 手动余弦 + exclude (目标域 bank 时屏蔽自身)
            full_sim = (e_q @ E_retr.t()).clamp(-1, 1)  # (B,N)
            bank_names = bank.get("names", None)
            if bank_names is not None:
                for b_idx in range(full_sim.shape[0]):
                    nm = batch_names[b_idx]
                    hit = [i for i, n in enumerate(bank_names) if n == nm]
                    if hit:
                        full_sim[b_idx, hit] = -float('inf')
            topv, topi = torch.topk(full_sim, k=args.k, dim=1)  # (B,k)
            idx, sim = topi, topv
            w = F.softmax(sim / args.rtemp, dim=1) if args.agg == "softmax" else sim
            P_fg, P_bg = aggregate(bank_gpu, idx, w, args.agg)
            prior = R.similarity_map(imge, P_fg, P_bg, args.tau)         # (B,1,32,32)
            gmap = None
            if args.gate == "hole":
                # 空洞 = base前景的 binary_fill_holes 后新增的背景像素(被前景包围的内腔)
                from scipy.ndimage import binary_fill_holes
                sb = (torch.sigmoid(base_logits)[:, 0] > 0.5).cpu().numpy()
                holes = np.stack([binary_fill_holes(s) & ~s for s in sb])   # (B,H,W) bool
                gmap = torch.from_numpy(holes).float().unsqueeze(1).to(base_logits.device)
            fused = R.fuse(base_logits, prior, args.lam, args.gate, args.gscale, gmap)   # (B,1,256,256)
        seg_base = (torch.sigmoid(base_logits)[:, 0] > 0.5).cpu().numpy()
        seg_rag = (torch.sigmoid(fused)[:, 0] > 0.5).cpu().numpy()
        gtn = gt.cpu().numpy()
        for j in range(seg_base.shape[0]):
            mb = R.seg_metrics(seg_base[j], gtn[j])
            mr = R.seg_metrics(seg_rag[j], gtn[j])
            base_arr.append(mb); rag_arr.append(mr)
            names.append(dp["image_name"][j] if isinstance(dp["image_name"], (list, tuple))
                         else str(dp["image_name"][j]))
        if bi % 10 == 0:
            print(f"  batch {bi}/{len(loader)} elapsed {time.time()-t0:.1f}s", flush=True)

    base_arr = np.array(base_arr); rag_arr = np.array(rag_arr)     # (N,6) dice,iou,acc,se,sp,hd
    keys = ["dice", "iou", "acc", "se", "sp", "hdis"]
    mb = base_arr.mean(0); mr = rag_arr.mean(0)
    rec = {
        "tag": (args.tag if args.tag else ("raghead" if head_on else "rag")),
        "ds": (args.ds_key or args.task), "ds_raw": args.task, "split": args.split,
        "k": args.k, "lambda": args.lam, "tau": args.tau,
        "retr": args.retr, "agg": args.agg, "stack_e": args.stack_e, "e_adapt": args.e_adapt,
        "gate": args.gate, "gscale": args.gscale,
        "n": int(len(names)),
        **{f"base_{kk}": float(mb[i]) for i, kk in enumerate(keys)},
        **{kk: float(mr[i]) for i, kk in enumerate(keys)},
        "dice_std": float(rag_arr[:, 0].std(ddof=1)),
        "per": {"base_dice": rag0(base_arr), "rag_dice": rag0(rag_arr),
                "base_se": se(base_arr), "rag_se": se(rag_arr), "names": names},
    }
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")

    if args.emit_table:
        emit_table_row(args.emit_table, rec, mr)

    d = mr - mb
    print(f"[rag] N={len(names)}  base dice={mb[0]:.2f} se={mb[3]:.2f} sp={mb[4]:.2f} hd={mb[5]:.2f}")
    print(f"[rag] RAG dice={mr[0]:.2f} se={mr[3]:.2f} sp={mr[4]:.2f} hd={mr[5]:.2f}")
    print(f"[rag] Δdice={d[0]:+.2f} Δse={d[3]:+.2f} Δsp={d[4]:+.2f}  -> {args.out}")
    paired_report(base_arr[:, 0], rag_arr[:, 0], base_arr[:, 3], rag_arr[:, 3])


def rag0(a): return [float(x) for x in a[:, 0]]
def se(a): return [float(x) for x in a[:, 3]]


def emit_table_row(path, rec, mr):
    """把 rag/raghead 行(键 (tag,ds)) upsert 进跨域总表 jsonl，字段对齐 ep270_300 行。"""
    keys = ["dice", "iou", "acc", "se", "sp", "hdis"]
    tag = rec.get("tag", "rag")
    ckpt_name = ("rag_pcr_head" if tag == "raghead" else
                 "rag_pcr_head_e" if tag in ("raghead_e", "raghead_e_tr2") else
                 "hf_adapt" if tag == "e_adapt" else
                 "hf_disk_e" if tag in ("hfal_e", "hfal_e6") else
                 "hf_disk_head_e" if tag in ("hfal_head", "hfal_head6", "hfal_head2") else
                 "shift50" if tag == "shift50" else
                 "multi_organ" if tag == "multi_tn3k" else
                 "bl_nothy" if tag == "bl_nothy" else "rag_fused")
    row = {"task": rec["ds"], "model": "SAMUS", "ckpt": ckpt_name,
           "tag": tag, "ds": rec["ds"],
           "params": {"k": rec["k"], "lambda": rec["lambda"], "tau": rec["tau"],
                      "retr": rec["retr"], "agg": rec["agg"], "stack_e": rec["stack_e"],
                      "e_adapt": rec.get("e_adapt")},
           "dice_std": rec["dice_std"], "base_dice": rec["base_dice"], "base_se": rec["base_se"],
           **{kk: float(mr[i]) for i, kk in enumerate(keys)}}
    rows = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("tag") == tag and d.get("ds") == rec["ds"]:
                continue
            rows.append(line)
    rows.append(json.dumps(row))
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"[rag] upsert tag={tag} ds={rec['ds']} -> {path}")


def paired_test(a, b):
    """配对 Wilcoxon(优先) 回退相关样本 t。返回 (stat, p, mean_diff)。"""
    diff = np.asarray(b) - np.asarray(a)
    md = float(diff.mean())
    try:
        from scipy.stats import wilcoxon
        s, p = wilcoxon(b, a, zero_method="wilcox")
        return float(s), float(p), md
    except Exception:
        n = len(diff); sd = diff.std(ddof=1)
        if sd == 0:
            return 0.0, 1.0, md
        t = md / (sd / np.sqrt(n))
        from math import erfc, sqrt
        p = erfc(abs(t) / sqrt(2))  # 近似(大样本正态)
        return float(t), float(p), md


def paired_report(bd, rd, bs, rs):
    for lbl, a, b in (("dice", bd, rd), ("se", bs, rs)):
        s, p, md = paired_test(a, b)
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        se_md = np.std(np.asarray(b) - np.asarray(a), ddof=1) / max(1, np.sqrt(len(a)))
        print(f"[paired] Δ{lbl}={md:+.2f} (SE {se_md:.2f}, |Δ|/SE={abs(md)/max(se_md,1e-9):.2f}) "
              f"Wilcoxon p={p:.4g} {star}")


if __name__ == "__main__":
    main()
