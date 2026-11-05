#!/usr/bin/env python3
# =============================================================================
# RAG(原型+检索) 共享件
#   目标：把 FUSE-RAG 的"ROI感知检索 + 原型条件化"两阶段，移植到我们冻结 SAMUS 的
#         source-only 零样本跨域(TN3K→DDTIz/PTz)设定，纯推理、不训练主干。
#   口径严格对齐 test.py/utils/evaluation.py:eval_mask_slice2：
#     - 前向 model(imgs) -> {'masks': B×1×256×256 logits}；seg = sigmoid>0.5
#     - gt 来自 datapack['label'][:,0]（二值前景，class_id=1）
#     - 指标用 utils.metrics.dice_coefficient / sespiou_coefficient2（输入 {0,255}），
#       hausdorff 曼哈顿距离；最后 ×100（百分值）
#   稠密特征：imge = model.image_encoder(imgs)，形状 (B, 256, 32, 32)（probe_rag.py 实测）。
#     原型与相似度图都在 32×32 网格上做，再双线性上采样到 256 与 base logits 融合。
# =============================================================================
import os, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from hausdorff import hausdorff_distance

from utils.config import get_config
from utils.data_us import JointTransform2D, ImageToImage2D
from models.model_dict import get_model
import utils.metrics as metrics

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = 32          # image_encoder 空间网格（imge 256×32×32）
DIM = 256          # 特征通道


# --------------------------------------------------------------------------- 模型
def build_model(task="DDTIz", load_path="", sam_ckpt="checkpoints/sam_vit_b_01ec64.pth",
                freq_adapter=0, c2f=0, c2f_iters=1, c2f_hidden=32):
    """与 test.py 完全一致的构建+权重加载。task 仅用于取 config(路径/data_path)。"""
    ns = argparse.Namespace(modelname="SAMUS", encoder_input_size=256, low_image_size=128,
                            task=task, vit_name="vit_b", sam_ckpt=sam_ckpt, batch_size=1, n_gpu=1,
                            base_lr=1e-4, warmup=False, warmup_period=250, keep_log=False,
                            device="cuda", load_path=load_path, freq_adapter=freq_adapter,
                            freq_adapter_share=0, c2f=c2f, c2f_iters=c2f_iters, c2f_hidden=c2f_hidden)
    opt = get_config(task)
    opt.mode = "val"; opt.visual = False; opt.modelname = "SAMUS"
    if load_path:
        opt.load_path = load_path
    ns.device = opt.device
    torch.manual_seed(300); np.random.seed(300)          # 与 test.py 同 seed
    torch.backends.cudnn.deterministic = True
    if freq_adapter:
        from models.segment_anything_samus.modeling.image_encoder import enable_freq_adapter
        enable_freq_adapter(share=False)
    if c2f:
        from models.segment_anything_samus.modeling.samus import enable_pbr
        enable_pbr(hidden=c2f_hidden, iters=c2f_iters)
    model = get_model("SAMUS", args=ns, opt=opt)
    ck = torch.load(opt.load_path, map_location="cpu")
    sd = {(k[7:] if k[:7] == "module." else k): v for k, v in ck.items()}
    model.load_state_dict(sd, strict=not (freq_adapter or c2f))
    model.to(opt.device); model.eval()
    return model, opt, ns


def make_loader(opt, split, batch_size=8):
    """确定性 val 式加载器：crop=None、增广概率全 0、p_freq=0 → 不污染记忆库。"""
    tf = JointTransform2D(img_size=256, low_img_size=128, ori_size=opt.img_size, crop=None,
                          p_flip=0, color_jitter_params=None, long_mask=True)
    ds = ImageToImage2D(opt.data_path, split, tf, img_size=256, class_id=1)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


# ------------------------------------------------------------------- 特征→原型
@torch.no_grad()
def dense_feat(model, imgs):
    """imge (B,256,32,32)"""
    return model.image_encoder(imgs.to(dtype=torch.float32))


def _mask_to_grid(mask):
    """(B,1,256,256) {0,1} → (B,32,32) 前景覆盖率 w∈[0,1]（平均池化，边界单元软化）。"""
    return F.avg_pool2d(mask.float(), kernel_size=256 // FEAT, stride=256 // FEAT)


def feat_and_proto(imge, mask):
    """给定稠密特征 imge(B,256,32,32) 与前景 mask(B,1,256,256)：
       返回 (retr_vec, P_fg, P_bg)
       - retr_vec(B,256): 检索用全局描述子（roi=True 时前景加权，见 retrieve 侧参数）
       - P_fg(B,256): 前景覆盖率加权平均特征（ROI-aware 支持嵌入=前景原型）
       - P_bg(B,256): 背景加权平均特征
       均 L2 归一化。"""
    B = imge.shape[0]
    locs = imge.flatten(2)                      # (B,256,1024)  列=空间位置
    locs = locs.transpose(1, 2)                 # (B,1024,256)
    w = _mask_to_grid(mask).flatten(1)          # (B,1024) 前景覆盖率
    w = w.clamp(0, 1)
    wbg = 1.0 - w
    def wmean(weights):
        s = weights.sum(1, keepdim=True).clamp_min(1e-6)
        return (locs * weights.unsqueeze(2)).sum(1) / s     # (B,256)
    P_fg = F.normalize(wmean(w), dim=1)
    P_bg = F.normalize(wmean(wbg), dim=1)
    glob = F.normalize(locs.mean(1), dim=1)     # 全局(所有位置)均值
    return glob, P_fg, P_bg


# ----------------------------------------------------------------------- 检索
def retrieve(e_q, E_bank, k):
    """暴力余弦 top-k。e_q(B,256), E_bank(N,256) → idx(B,k), sim(B,k)。"""
    sim = e_q @ E_bank.t()                      # 已 L2norm → 余弦
    sim = sim.clamp(-1, 1)
    topv, topi = torch.topk(sim, k=k, dim=1)
    return topi, topv


# ------------------------------------------------- 相似度先验图 + 融合
def similarity_map(imge_q, P_fg, P_bg, tau):
    """逐位置与聚合前景/背景原型的余弦差 → (B,1,32,32) logit 先验。
       imge_q(B,256,32,32)，P_fg/P_bg(B,256)。"""
    B = imge_q.shape[0]
    locs = imge_q.flatten(2).transpose(1, 2)             # (B,1024,256)
    locs = F.normalize(locs, dim=2)
    sfg = (locs * P_fg.unsqueeze(1)).sum(2)              # (B,1024)
    sbg = (locs * P_bg.unsqueeze(1)).sum(2)
    prior = tau * (sfg - sbg)                             # logit 尺度
    return prior.view(B, 1, FEAT, FEAT)


def fuse(base_logits, prior32, lam, gate="off", gscale=4.0, gmap=None):
    """base_logits(B,1,256,256) + lam * 上采样先验(B,1,32,32→256)。
       gate="conf"：置信度门控 g=exp(-|base_logits|/gscale)∈(0,1]，只在 SAMUS
       低置信区注入先验，高置信区保持原判。
       gmap 不为 None 时优先用之：逐像素门控张量(B,1,256,256)∈[0,1]，先验只在该区域
       生效(如 hole-filling 的空洞区)。gate="off" 即全局均匀加性。"""
    prior = F.interpolate(prior32, size=base_logits.shape[-2:], mode="bilinear", align_corners=False)
    if gmap is not None:
        return base_logits + lam * gmap * prior
    if gate == "conf":
        g = torch.exp(-(base_logits.abs() / max(gscale, 1e-6)))
        return base_logits + lam * g * prior
    return base_logits + lam * prior


# -------------------------------------------------- PCR 头专用：检索→聚合→逐位置余弦
def prototypes_for_query(imge, bank, k, exclude=None, retr="roi", rtemp=0.1):
    """由 query 稠密特征 imge(B,256,FEAT,FEAT) 检索源域原型，返回逐位置与聚合前景/背景
       原型的余弦图 sim_fg/sim_bg (B,1,FEAT,FEAT)，供 PCRHead 条件化使用。
       逻辑与 rag_segment 主路一致（非对称 query=全局均值向量→暴力余弦 top-k→softmax 加权
       聚合 P_fg(=E_roi)/P_bg），统一到公共函数。
       exclude: list[str] 长度 B——训练时按 image_name 命中 bank names 的自身条目屏蔽
                (leave-one-out)，杜绝同图自检索泄漏。跨域测试时 query 名不在库内→自然 no-op。"""
    device = imge.device
    E_retr = bank["E_roi" if retr == "roi" else "E_mean"].to(device)
    E_fg = bank["E_roi"].to(device)          # ROI 加权嵌入 == 前景原型 p_fg
    P_bg_bank = bank["P_bg"].to(device)
    locs = imge.flatten(2).transpose(1, 2)   # (B,FEAT*FEAT,256)
    e_q = F.normalize(locs.mean(1), dim=1)   # query 全局均值检索向量
    sim = (e_q @ E_retr.t()).clamp(-1, 1)    # (B,N) 余弦
    if exclude is not None:
        names = bank.get("names", None)
        if names is not None:
            for b in range(sim.shape[0]):
                nm = exclude[b] if isinstance(exclude, (list, tuple)) else None
                if nm is None:
                    continue
                hit = [i for i, x in enumerate(names) if x == nm]
                if hit:
                    sim[b, hit] = float("-inf")
    topv, topi = torch.topk(sim, k=k, dim=1)                 # (B,k)
    w = F.softmax(topv / max(rtemp, 1e-6), dim=1)
    P_fg = F.normalize(torch.bmm(w.unsqueeze(1), E_fg[topi]).squeeze(1), dim=1)   # (B,256)
    P_bg = F.normalize(torch.bmm(w.unsqueeze(1), P_bg_bank[topi]).squeeze(1), dim=1)
    locn = F.normalize(locs, dim=2)
    sfg = (locn * P_fg.unsqueeze(1)).sum(2).view(-1, 1, FEAT, FEAT)
    sbg = (locn * P_bg.unsqueeze(1)).sum(2).view(-1, 1, FEAT, FEAT)
    return sfg, sbg


def _gaussian_kernel(sigma, ksize, device, dtype):
    coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2 = (g[:, None] @ g[None, :])
    return k2.view(1, 1, ksize, ksize)


def shift_image(imgs, strength=1.0):
    """训练时把 query 图退化为「跨域感」，迫使 PCR 头依赖检索先验(而非走域内捷径)。
       纯张量、mask 无关；imgs(B,C,H,W)∈[0,1]（超声多为 C=1）。逐图随机组合四种退化：
       对比度抖动 / 高斯噪声 / 高斯模糊 / 中高频幅度衰减(低通)。strength 缩放幅度。"""
    if strength <= 0:
        return imgs
    dev, dt = imgs.device, imgs.dtype
    B, C = imgs.shape[0], imgs.shape[1]
    out = imgs
    # 1) 对比度：围绕每图均值缩放 c∈[1-0.4s, 1+0.4s]
    c = 1.0 + strength * (torch.rand(B, 1, 1, 1, device=dev, dtype=dt) * 2 - 1) * 0.4
    mean = out.mean(dim=(1, 2, 3), keepdim=True)
    out = ((out - mean) * c + mean).clamp(0, 1)
    # 2) 高斯噪声：σ∈[0, 0.12s]
    sig = strength * torch.rand(B, 1, 1, 1, device=dev, dtype=dt) * 0.12
    out = (out + torch.randn_like(out) * sig).clamp(0, 1)
    # 3) 高斯模糊：逐图以 ~50% 概率做 5×5 低通
    blur = torch.rand(B, device=dev) < 0.5
    if bool(blur.any()):
        ksz = 5
        k = _gaussian_kernel(1.5 * strength, ksz, dev, dt).expand(C, 1, ksz, ksz)
        blurred = F.conv2d(out, k, padding=ksz // 2, groups=C).clamp(0, 1)
        out = torch.where(blur.view(B, 1, 1, 1), blurred, out)
    # 4) 中高频幅度衰减：fft 后按半径 r∈[0.25,1] 的升序权重线性压低高频(与方案E反向)
    H, W = out.shape[-2:]
    Fc = torch.fft.fftshift(torch.fft.fft2(out, dim=(-2, -1)), dim=(-2, -1))
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H, device=dev, dtype=dt),
        torch.linspace(-1, 1, W, device=dev, dtype=dt), indexing="ij")
    r = torch.sqrt((xs * 0.5) ** 2 + (ys * 0.5) ** 2)
    attn = strength * torch.rand(B, 1, 1, 1, device=dev, dtype=dt)      # 每图衰减强度∈[0,s]
    hi = (r >= 0.25 * 0.707).view(1, 1, H, W)                          # bool 高通掩码
    amp = Fc.abs()
    amp = torch.where(hi, amp * (1.0 - attn).clamp_min(0.0), amp)
    out = torch.fft.ifft2(torch.fft.ifftshift(amp * torch.exp(1j * Fc.angle()), dim=(-2, -1)),
                          dim=(-2, -1)).real.clamp(0, 1)
    return out.to(dt)


# ------------------------------------------------------------------- 指标(对齐 eval_mask_slice2)
def seg_metrics(seg, gt):
    """seg/gt: (h,w) 二值(bool)。返回百分值 dice,iou,acc,se,sp,hd95。"""
    h, w = seg.shape
    pred_i = np.zeros((1, h, w)); pred_i[0][seg] = 255
    gt_i = np.zeros((1, h, w)); gt_i[0][gt] = 255
    dice = metrics.dice_coefficient(pred_i, gt_i)
    iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
    hd = hausdorff_distance(pred_i[0], gt_i[0], distance="manhattan")
    return [dice * 100, iou * 100, acc * 100, se * 100, sp * 100, float(hd)]


@torch.no_grad()
def model_base_logits(model, imgs):
    out = model(imgs.to(dtype=torch.float32))
    return out["masks"]
