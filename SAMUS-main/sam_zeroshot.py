#!/usr/bin/env python3
# =============================================================================
# sam_zeroshot.py — 归因消融：完全不训 TN3K 的原生 SAM + 原型/RAG(TN3K train 当检索库)
#   目的：把「源域微调」的贡献从当前成绩里剥离出来。之前所有零样本行(ep270/base50/
#         方案E/head×E)都内嵌了 TN3K 微调过的 encoder。这里 encoder 换成**未微调的
#         原生 SAM ViT-B**（sam_vit_b，SA-1B 预训练、没碰过甲状腺），只把 TN3K train
#         当作原型/检索参考库（推理侧、无训练），看 DDTI/PT 能到多少 → 与微调链之差
#         ≈ TN3K 训练买到的 dice；与 naive 候选选掩码之差 ≈ 检索机制买到的 dice。
#   口径：真·零训练(无任何梯度)；源域仅出现在「建原型库」这一步(推理侧材料)，与
#         rag_bank 零样本口径一致；评测严格复用 rag_common.make_loader + seg_metrics
#         (=test.py/eval_mask_slice2 同尺)，per-image 可与 base50/ep270 按 image_name 配对。
#   两种预测(两种 tag)：
#     --mode map : 稠密原型最近邻。query 每位置与检索到的前景/背景原型余弦差→64×64 先验
#                  →上采样 256→(>thr) 二值。最鲁棒、纯原型对比。
#     --mode amg : 原生 SAM AMG 出无类别候选→每个候选区域嵌入 vs 甲状腺原型打分→取最优。
#                  即「AMG+原型选掩码」，用户选定路径。
#   库：rag_bank/sam_tn3k_bank.pt（E_roi/E_mean/P_bg 各 (N,256)+names），原生 SAM 特征空间。
# =============================================================================
import os, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import rag_common as R
from models.segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

HERE = R.HERE
SAM_CKPT = "/home/linux/Downloads/wsp/KTD/sam_vit_b_01ec64.pth"
G = 64           # 原生 SAM neck 空间网格（256→1024 输入，1024/16=64）
SRC = 256        # loader 输出分辨率（方图 → 无 padding，neck 均匀映射回 256）


def load_sam(ckpt, device):
    sam = sam_model_registry["vit_b"](checkpoint=ckpt)
    sam.eval()
    return sam.to(device)


def to_np_rgb(img_chw):
    """(256,256) float[0,1] → HWC uint8 RGB（SAM 自带 transform 会 resize+归一化，不可二次归一）。"""
    a = (np.asarray(img_chw) * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([a] * 3, axis=-1)


def feats_of(predictor, np_rgb):
    predictor.set_image(np_rgb)                 # 内部完成 resize→1024、归一化、编码
    return predictor.features[0].float()        # (256,64,64)


def mask_to_grid(m_bool, dev):
    """(256,256) bool → (64,64) 前景覆盖率∈[0,1]（avg_pool 因子 4）。"""
    t = torch.from_numpy(m_bool.astype(np.float32)).view(1, 1, SRC, SRC).to(dev)
    return F.avg_pool2d(t, kernel_size=SRC // G, stride=SRC // G)[0, 0]


def loc_vecs(feats):
    """(256,64,64) → locs(4096,256 原始) 与 locn(L2 归一)。flatten 行主序，与网格一致。"""
    locs = feats.flatten(1).t()                 # (4096,256)
    return locs, F.normalize(locs, dim=1)


def compute_proto(feats, mask_bool, dev):
    """一图源：返回 (e_global(256) 检索描述子, P_fg(256), P_bg(256))，均 L2 归一。"""
    locs, _ = loc_vecs(feats)
    w = mask_to_grid(mask_bool, dev).flatten().clamp(0, 1)
    def wmean(a):
        s = a.sum().clamp_min(1e-6)
        return (locs * a[:, None]).sum(0) / s
    P_fg = F.normalize(wmean(w), dim=0)
    P_bg = F.normalize(wmean(1.0 - w), dim=0)
    e_g = F.normalize(feats.mean(dim=(1, 2)), dim=0)
    return e_g, P_fg, P_bg


# ------------------------------------------------------------------- 建库
@torch.no_grad()
def build_bank(sam, task, split, device, out_path, batch=8, limit=0):
    opt = R.get_config(task)
    loader = R.make_loader(opt, split, batch_size=batch)
    predictor = SamPredictor(sam)
    E_roi, E_mean, P_bg, names = [], [], [], []
    t0 = time.time()
    for bi, dp in enumerate(loader):
        imgs = dp["image"]
        gts = (dp["label"][:, 0] == 1).cpu().numpy()
        for j in range(imgs.shape[0]):
            np_rgb = to_np_rgb(imgs[j, 0].numpy())
            feats = feats_of(predictor, np_rgb).to(device)
            eg, pf, pb = compute_proto(feats, gts[j], device)
            E_roi.append(pf.cpu()); E_mean.append(eg.cpu()); P_bg.append(pb.cpu())
            names.append(dp["image_name"][j] if isinstance(dp["image_name"], (list, tuple))
                         else str(dp["image_name"][j]))
        if bi % 20 == 0:
            print(f"  [bank] {bi+1}/{len(loader)} elapsed {time.time()-t0:.1f}s", flush=True)
        if limit and len(names) >= limit:
            print(f"  [bank] 达 limit={limit}，提前停止(冒烟用)", flush=True)
            break
    bank = {"E_roi": torch.stack(E_roi), "E_mean": torch.stack(E_mean),
            "P_bg": torch.stack(P_bg), "names": names,
            "meta": {"src": f"{task}/{split}", "dim": 256, "grid": G, "enc": "raw_sam_vit_b"}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(bank, out_path)
    print(f"[bank] built N={len(names)} dim=256 grid={G} -> {out_path}  ({time.time()-t0:.1f}s)", flush=True)


# ------------------------------------------------------------------- 检索+聚合
def retrieve_aggregate(bank_gpu, e_q, k, rtemp):
    E_mean = bank_gpu["E_mean"]; E_roi = bank_gpu["E_roi"]; P_bg = bank_gpu["P_bg"]
    sim = (e_q @ E_mean.t()).clamp(-1, 1)                 # (1,N)
    topv, topi = torch.topk(sim, k=k, dim=1)
    w = F.softmax(topv / max(rtemp, 1e-6), dim=1)         # (1,k)
    P_fg = F.normalize(torch.bmm(w.unsqueeze(1), E_roi[topi]).squeeze(1), dim=1)   # (1,256)
    P_bg = F.normalize(torch.bmm(w.unsqueeze(1), P_bg[topi]).squeeze(1), dim=1)
    return P_fg, P_bg


# ------------------------------------------------------------------- 预测
def predict_map(feats, bank_gpu, e_q, k, rtemp, thr, device):
    prior_up = map_prior_up(feats, bank_gpu, e_q, k, rtemp, device)
    return (prior_up > thr)


def map_prior_up(feats, bank_gpu, e_q, k, rtemp, device):
    """返回上采样到 256 的余弦差先验图 (256,256) float numpy（未阈化）。"""
    _, locn = loc_vecs(feats)
    P_fg, P_bg = retrieve_aggregate(bank_gpu, e_q, k, rtemp)
    sfg = (locn * P_fg).sum(1)
    sbg = (locn * P_bg).sum(1)
    prior = (sfg - sbg).view(1, 1, G, G)
    prior_up = F.interpolate(prior, size=(SRC, SRC), mode="bilinear", align_corners=False)
    return prior_up[0, 0].cpu().numpy()


def predict_amg(amg, np_rgb, feats, bank_gpu, e_q, k, rtemp, device,
                min_frac=0.002, max_frac=0.30, merge_margin=0.02):
    """AMG 无类别候选 → 按结节合理面积区间筛(避免碎块/整图) → 与甲状腺原型打分取最优。
       merge_margin>0 时把与最优分数差距在此内的候选并起来(结节常被切成多块)。"""
    masks = amg.generate(np_rgb)
    if not masks:
        return np.zeros((SRC, SRC), bool)
    locs, _ = loc_vecs(feats)
    P_fg, P_bg = retrieve_aggregate(bank_gpu, e_q, k, rtemp)
    pfg, pb = P_fg[0], P_bg[0]
    total = float(SRC * SRC)
    cand = []
    for m in masks:
        seg = m["segmentation"]
        if seg.shape != (SRC, SRC):
            import cv2
            seg = cv2.resize(seg.astype(np.uint8), (SRC, SRC), interpolation=cv2.INTER_NEAREST) > 0.5
        frac = float(seg.sum()) / total
        if frac < min_frac or frac > max_frac:            # 面积闸门：滤碎块与整图
            continue
        cov = mask_to_grid(seg, device).flatten()
        region = F.normalize((locs * cov[:, None]).sum(0) / cov.sum().clamp_min(1e-6), dim=0)
        cand.append((float(region @ pfg - region @ pb), seg))
    if not cand:
        return np.zeros((SRC, SRC), bool)
    cand.sort(key=lambda t: -t[0])
    best_s = cand[0][0]
    out = cand[0][1].copy()
    if merge_margin and merge_margin > 0:
        for s, seg in cand[1:]:
            if s >= best_s - merge_margin:
                out |= seg
            else:
                break
    return out


# ------------------------------------------------------------------- 源域 val 标定 map 阈值（无泄漏）
@torch.no_grad()
def calibrate_thr(sam, bank_gpu, args, device, grid=None):
    """在源域(bank_task/bank_split 同域) val 上扫 thr 使 dice 最大，返回最优 thr。
       只在源域选参，绝不碰 DDTIz/PTz（计局对齐 rag_select 纪律）。"""
    if grid is None:
        grid = np.linspace(-0.1, 0.6, 29)
    opt = R.get_config(args.cal_task)
    split = "val"
    loader = R.make_loader(opt, split, batch_size=1)
    predictor = SamPredictor(sam)
    priors, gts = [], []
    for bi, dp in enumerate(loader):
        np_rgb = to_np_rgb(dp["image"][0, 0].numpy())
        gt = (dp["label"][0, 0] == 1).cpu().numpy()
        if not gt.any():
            continue
        feats = feats_of(predictor, np_rgb).to(device)
        e_q = F.normalize(feats.mean(dim=(1, 2)), dim=0).unsqueeze(0)
        priors.append(map_prior_up(feats, bank_gpu, e_q, args.k, args.rtemp, device).astype(np.float16))
        gts.append(gt)
        if bi >= args.cal_max - 1:
            break
    best_thr, best_d = float(grid[0]), -1.0
    for thr in grid:
        ds = []
        for p, g in zip(priors, gts):
            seg = (p.astype(np.float32) > thr)
            inter = np.logical_and(seg, g).sum(); s = seg.sum() + g.sum()
            ds.append((2 * inter / s) if s else 0.0)
        md = float(np.mean(ds)) if ds else 0.0
        if md > best_d:
            best_thr, best_d = float(thr), md
    print(f"[zs] calibrate thr on {args.cal_task}/{split} (n={len(gts)}): best thr={best_thr:.3f} val dice={best_d:.2f}", flush=True)
    return best_thr


# ------------------------------------------------------------------- 主评测
@torch.no_grad()
def run_eval(sam, bank_gpu, args, device):
    opt = R.get_config(args.task)
    split = opt.test_split if args.split == "test" else opt.val_split
    if args.task == "TN3K":
        split = "val" if args.split == "val" else opt.test_split
    loader = R.make_loader(opt, split, batch_size=1)
    predictor = SamPredictor(sam)
    amg = None
    if args.mode == "amg":
        amg = SamAutomaticMaskGenerator(sam, points_per_side=args.pts, pred_iou_thresh=0.7,
                                        stability_score_thresh=0.5)
    names, mets = [], []
    t0 = time.time()
    for bi, dp in enumerate(loader):
        np_rgb = to_np_rgb(dp["image"][0, 0].numpy())
        feats = feats_of(predictor, np_rgb).to(device)
        gt = (dp["label"][0, 0] == 1).cpu().numpy()
        e_q = F.normalize(feats.mean(dim=(1, 2)), dim=0).unsqueeze(0)   # 检索描述子=全局均值, 不看 GT
        if args.mode == "map":
            seg = predict_map(feats, bank_gpu, e_q, args.k, args.rtemp, args.thr, device)
        else:
            seg = predict_amg(amg, np_rgb, feats, bank_gpu, e_q, args.k, args.rtemp, device,
                              args.amg_min_frac, args.amg_max_frac, args.amg_merge)
        mm = R.seg_metrics(seg, gt)                     # [dice,iou,acc,se,sp,hd]
        mets.append(mm)
        names.append(dp["image_name"][0] if isinstance(dp["image_name"], (list, tuple)) else str(dp["image_name"]))
        if bi % 20 == 0:
            print(f"  [{args.mode}] {bi+1}/{len(loader)} dice={mm[0]:.2f} elapsed {time.time()-t0:.1f}s", flush=True)
    arr = np.array(mets)                                 # (N,6)
    mean = arr.mean(0)
    keys = ["dice", "iou", "acc", "se", "sp", "hdis"]
    rec = {"tag": args.tag, "ds": (args.ds_key or args.task), "ds_raw": args.task,
           "mode": args.mode, "split": split, "k": args.k, "rtemp": args.rtemp, "thr": args.thr,
           "n": int(len(names)),
           **{kk: float(mean[i]) for i, kk in enumerate(keys)},
           "dice_std": float(arr[:, 0].std(ddof=1)),
           "per": {"dice": [float(x) for x in arr[:, 0]], "se": [float(x) for x in arr[:, 3]],
                   "rag_dice": [float(x) for x in arr[:, 0]], "rag_se": [float(x) for x in arr[:, 3]],
                   "names": names}}
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if args.emit_table:
        emit_row(args.emit_table, rec)
    print(f"[zs:{args.mode}] {args.task} N={len(names)}  dice={mean[0]:.2f} se={mean[3]:.2f} "
          f"sp={mean[4]:.2f} hd={mean[5]:.2f}  -> {args.out}", flush=True)


def emit_row(path, rec):
    row = {"task": rec["ds"], "model": "SAM", "ckpt": "raw_sam_vit_b",
           "tag": rec["tag"], "ds": rec["ds"],
           "params": {"mode": rec["mode"], "k": rec["k"], "rtemp": rec["rtemp"], "thr": rec["thr"]},
           "dice_std": rec["dice_std"], "base_dice": None, "base_se": None,
           "dice": rec["dice"], "iou": rec["iou"], "acc": rec["acc"],
           "se": rec["se"], "sp": rec["sp"], "hdis": rec["hdis"]}
    tag, ds = rec["tag"], rec["ds"]
    rows = []
    if os.path.exists(path):
        for line in open(path):
            s = line.strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except Exception:
                continue
            if d.get("tag") == tag and d.get("ds") == ds:
                continue
            rows.append(s)
    rows.append(json.dumps(row))
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"[zs] upsert tag={tag} ds={ds} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam-ckpt", default=SAM_CKPT)
    ap.add_argument("--bank", default=os.path.join(HERE, "rag_bank", "sam_tn3k_bank.pt"))
    ap.add_argument("--build-bank", dest="build_bank", action="store_true")
    ap.add_argument("--bank-task", default="TN3K"); ap.add_argument("--bank-split", default="train")
    ap.add_argument("--bank-limit", dest="bank_limit", type=int, default=0, help=">0 时只建这么多个原型(冒烟用)")
    ap.add_argument("--task", default="DDTIz", help="DDTIz|PTz|TN3K")
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", choices=["map", "amg"], default="map")
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--rtemp", type=float, default=0.1)
    ap.add_argument("--thr", type=float, default=0.0, help="map 模式余弦差阈值(0=sign)")
    ap.add_argument("--pts", type=int, default=16, help="AMG points_per_side")
    ap.add_argument("--amg-min-frac", dest="amg_min_frac", type=float, default=0.002, help="AMG 候选最小面积占比(滤碎块)")
    ap.add_argument("--amg-max-frac", dest="amg_max_frac", type=float, default=0.30, help="AMG 候选最大面积占比(滤整图)")
    ap.add_argument("--amg-merge", dest="amg_merge", type=float, default=0.02, help="AMG 合并分数容差(0=不并)")
    ap.add_argument("--calibrate", dest="calibrate", action="store_true", help="map 阈值在源域 val 标定(无泄漏)")
    ap.add_argument("--cal-task", dest="cal_task", default="TN3K")
    ap.add_argument("--cal-max", dest="cal_max", type=int, default=499, help="标定最多扫描的 val 图数")
    ap.add_argument("--out", default=os.path.join(HERE, "sam_zeroshot_results.jsonl"))
    ap.add_argument("--no-eval", dest="no_eval", action="store_true", help="只建库(--build-bank)后退出，不评测")
    ap.add_argument("--emit-table", dest="emit_table", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ds-key", dest="ds_key", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = load_sam(args.sam_ckpt, device)
    if args.build_bank:
        build_bank(sam, args.bank_task, args.bank_split, device, args.bank, limit=args.bank_limit)
    if args.no_eval or not args.task:
        return
    bank = torch.load(args.bank, map_location="cpu")
    bank_gpu = {kk: bank[kk].to(device) for kk in ["E_roi", "E_mean", "P_bg"]}
    if not args.tag:
        args.tag = "samzs_map" if args.mode == "map" else "samzs_amg"
    if args.calibrate and args.mode == "map":
        args.thr = calibrate_thr(sam, bank_gpu, args, device)
    run_eval(sam, bank_gpu, args, device)


if __name__ == "__main__":
    main()
