#!/usr/bin/env python3
"""一次性维度探测：载入与 test.py 完全相同的 SAMUS 模型，跑一张图打印
image_encoder 稠密特征 imge 的形状/通道、pred['masks'] 形状、gt label 形状。
锁定 RAG 原型/相似度图所需的空间网格尺寸后，此脚本可删。"""
import os, argparse, torch, numpy as np
from torch.utils.data import DataLoader
from utils.config import get_config
from utils.data_us import JointTransform2D, ImageToImage2D
from models.model_dict import get_model

torch.backends.cudnn.deterministic = True

def build(task, sam_ckpt="checkpoints/sam_vit_b_01ec64.pth"):
    ns = argparse.Namespace(modelname="SAMUS", encoder_input_size=256, low_image_size=128,
                            task=task, vit_name="vit_b", sam_ckpt=sam_ckpt, batch_size=1, n_gpu=1,
                            base_lr=1e-4, warmup=False, warmup_period=250, keep_log=False,
                            device="cuda", load_path="", freq_adapter=0, freq_adapter_share=0,
                            c2f=0, c2f_iters=1, c2f_hidden=32)
    opt = get_config(task)
    opt.mode = "val"; opt.visual = False; opt.modelname = "SAMUS"
    ns.device = opt.device
    tf = JointTransform2D(img_size=256, low_img_size=128, ori_size=opt.img_size, crop=None,
                          p_flip=0, color_jitter_params=None, long_mask=True)
    ds = ImageToImage2D(opt.data_path, opt.test_split, tf, img_size=256, class_id=1)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    model = get_model("SAMUS", args=ns, opt=opt)
    ck = torch.load(opt.load_path)
    sd = {(k[7:] if k[:7] == "module." else k): v for k, v in ck.items()}
    model.load_state_dict(sd, strict=True)
    model.to(opt.device); model.eval()
    return model, ds, dl, opt

if __name__ == "__main__":
    model, ds, dl, opt = build("DDTIz")
    print("num images in split:", len(ds))
    dp = next(iter(dl))
    imgs = dp["image"].to(device=opt.device, dtype=torch.float32)
    with torch.no_grad():
        imge = model.image_encoder(imgs)
        out = model(imgs)
    print("imgs", tuple(imgs.shape))
    print("imge(image_encoder)", tuple(imge.shape), "dtype", imge.dtype)
    print("pred['masks']", tuple(out["masks"].shape))
    print("low_res_logits", tuple(out["low_res_logits"].shape))
    print("label", tuple(dp["label"].shape), "uniq", torch.unique(dp["label"]).tolist())
    fg = (dp["label"][:, 0] == 1).float()
    print("gt foreground pixels:", int(fg.sum().item()))
