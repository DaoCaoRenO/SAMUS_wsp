#!/usr/bin/env python3
# =============================================================================
# rag_build_bank.py — 构建并缓存 TN3K 源域记忆库（一次性、纯推理、不训练）
#   对应计划 §2。用冻结 SAMUS(ep270) 的 image_encoder 稠密特征 imge(B,256,32,32)，
#   对 TN3K `train`(1999) 每张已标注图提取：
#     - E_roi  = L2norm(前景覆盖率加权池化)  → ROI-aware 支持检索嵌入(=论文 e_s)，同时作前景原型 p_fg
#     - E_mean = L2norm(全位置均值池化)      → 消融用「非ROI均值检索」备选向量
#     - P_bg   = L2norm(背景加权池化)        → 背景原型 p_bg
#   口径全部走 rag_common（与 test.py/eval_mask_slice2 一致），确定性 val 式加载、seed=300。
#   产物：rag_bank/tn3k_bank.pt  (dict: E_roi/E_mean/P_bg/names/feat/dim)
# 用法：python3 rag_build_bank.py [--task TN3K] [--ckpt ...] [--batch 16] [--out rag_bank/tn3k_bank.pt]
# =============================================================================
import os, argparse, time
import torch
import rag_common as R


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="TN3K", help="记忆库来源任务(源域)")
    ap.add_argument("--ckpt", default="checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth")
    ap.add_argument("--split", default="train", help="源域切分(默认 train=1999)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(R.HERE, "rag_bank", "tn3k_bank.pt"))
    args = ap.parse_args()

    model, opt, ns = R.build_model(task=args.task, load_path=args.ckpt)
    loader = R.make_loader(opt, args.split, batch_size=args.batch)

    E_roi, E_mean, P_bg, names = [], [], [], []
    t0 = time.time()
    for bi, dp in enumerate(loader):
        imgs = dp["image"].to(opt.device, dtype=torch.float32)
        # 前景二值 mask (B,1,256,256)：与 eval_mask_slice2 的 gt 完全同源(label[:,0])
        mask = dp["label"].to(opt.device, dtype=torch.float32)
        imge = R.dense_feat(model, imgs)                 # (B,256,32,32)
        glob, pf, pb = R.feat_and_proto(imge, mask)      # 均 L2norm(B,256)
        E_mean.append(glob.float().cpu())
        E_roi.append(pf.float().cpu())                   # ROI-aware 检索嵌入 == 前景原型
        P_bg.append(pb.float().cpu())
        names.extend(list(dp["image_name"]))
        if bi % 20 == 0:
            print(f"  batch {bi}/{len(loader)} elapsed {time.time()-t0:.1f}s", flush=True)

    bank = {
        "E_roi": torch.cat(E_roi),       # (N,256) 检索用(ROI) + 前景原型 p_fg
        "E_mean": torch.cat(E_mean),     # (N,256) 检索用(均值) 消融备选
        "P_bg": torch.cat(P_bg),         # (N,256) 背景原型 p_bg
        "names": names,
        "feat": R.FEAT, "dim": R.DIM,
        "ckpt": os.path.basename(args.ckpt), "task": args.task, "split": args.split,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(bank, args.out)
    N = bank["E_roi"].shape[0]
    print(f"[build_bank] saved N={N} E_roi{tuple(bank['E_roi'].shape)} "
          f"E_mean{tuple(bank['E_mean'].shape)} P_bg{tuple(bank['P_bg'].shape)} -> {args.out}")
    print(f"[build_bank] {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
