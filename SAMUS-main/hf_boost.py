#!/usr/bin/env python3
"""
方案E：测试时高频幅度增益（zero-training 频域补偿）

动机（可测量、非拍脑袋）：三个数据集原始分辨率差异巨大 ——
    TN3K 402×314、DDTI 360×560、private_Thyroid 697×876，
而管线统一 resize 到 256。降采样本身是低通滤波：PT 被降采样 ~3.4×、TN3K 仅 ~1.4×，
因此**目标域高频被过度衰减**，与源域(TN3K)训练时见到的频谱统计不一致。
本脚本对 r>=ratio 的高频幅度乘以 gain（相位不变 → 内容/位置不变，GT 原样沿用），
把目标域高频拉回源域量级。gain=1.0 即对照组（仅多一次 JPEG 重编码）。

严格复现管线的 resize：cv2 读取 -> F.to_pil_image -> F.resize(256, BILINEAR)，
与 utils/data_us.py:237 完全一致，因此对照组可用于验证整条链路可信度。
"""
import os, json, argparse
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "dataset", "SAMUS")
SIZE = 256
# tag -> (源 split 文件名, 源子目录)
TASKS = {
    "TN3K":  ("test",         "TN3K"),
    "DDTIz": ("test_DDTIz",   "DDTIz"),
    "PTz":   ("test_PTz",     "PTz"),
}


def load_resized(sub, fname):
    """按管线同款方式读取并 resize 到 256（返回 float32 [0,1]）"""
    p = os.path.join(DS, sub, "img", fname + ".jpg")
    im = cv2.imread(p, 0)
    if im is None:
        return None
    pil = TF.to_pil_image(im)
    pil = TF.resize(pil, (SIZE, SIZE), InterpolationMode.BILINEAR)
    return np.asarray(pil).astype(np.float32)


def hf_mask(H, W, ratio):
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H // 2, W // 2
    r = np.sqrt(((yy - cy) / (H / 2.0)) ** 2 + ((xx - cx) / (W / 2.0)) ** 2)
    return r >= ratio


def hf_energy(img_f, mask):
    """高频平均幅度（用于测量源/目标域的频谱失配程度）"""
    F = np.fft.fftshift(np.fft.fft2(img_f))
    return float(np.abs(F)[mask].mean())


def boost(img_f, mask, gain):
    F = np.fft.fftshift(np.fft.fft2(img_f))
    amp, pha = np.abs(F), np.angle(F)
    amp = amp.copy()
    amp[mask] *= gain
    out = np.real(np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha))))
    return np.clip(out, 0, 255).astype(np.uint8)


def build(tag, gain, ratio, mask):
    split, sub = TASKS[tag]
    lines = [l.strip() for l in open(os.path.join(DS, "MainPatient", split + ".txt")) if l.strip()]
    out_root = os.path.join(DS, f"{tag}_hf")
    for d in ("img", "label"):
        os.makedirs(os.path.join(out_root, d), exist_ok=True)
    out_lines, n = [], 0
    for ln in lines:
        parts = ln.split("/")
        s_sub, fname, cls = parts[0], parts[1], parts[2]
        im = load_resized(s_sub, fname)
        if im is None:
            continue
        cv2.imwrite(os.path.join(out_root, "img", fname + ".jpg"), boost(im, mask, gain))
        link = os.path.join(out_root, "label", fname + ".jpg")
        if not os.path.lexists(link):
            os.symlink(os.path.join(DS, s_sub, "label", fname + ".jpg"), link)
        out_lines.append(f"{tag}_hf/{fname}/{cls}")
        n += 1
    with open(os.path.join(DS, "MainPatient", f"test_{tag}_hf.txt"), "w") as f:
        f.write("\n".join(out_lines) + "\n")
    # 注册 class.json（沿用源 tag 的类别数，保证 mask 二值化逻辑一致）
    cjp = os.path.join(DS, "MainPatient", "class.json")
    cj = json.load(open(cjp))
    cj[f"{tag}_hf"] = cj.get(sub, 2)
    json.dump(cj, open(cjp, "w"))
    return n


def measure(ratio, n_sample=120):
    """测量各数据集 256 尺度下的高频平均幅度，给出有依据的 γ* = 源域/目标域"""
    mask = hf_mask(SIZE, SIZE, ratio)
    rng = np.random.RandomState(0)
    res = {}
    for tag, (split, sub) in TASKS.items():
        lines = [l.strip() for l in open(os.path.join(DS, "MainPatient", split + ".txt")) if l.strip()]
        pick = rng.choice(len(lines), size=min(n_sample, len(lines)), replace=False)
        e = []
        for i in pick:
            parts = lines[i].split("/")
            im = load_resized(parts[0], parts[1])
            if im is not None:
                e.append(hf_energy(im, mask))
        res[tag] = float(np.mean(e))
    print("高频平均幅度(r>=%.2f, 256尺度):" % ratio)
    for t, v in res.items():
        star = res["TN3K"] / v if v > 0 else float("nan")
        print("  %-6s %10.2f   γ*(=TN3K/该集) = %.3f" % (t, v, star))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, default=1.0, help="高频幅度增益；1.0=对照组")
    ap.add_argument("--ratio", type=float, default=0.25, help="高频半径阈值(归一化 r>=ratio 视为高频)")
    ap.add_argument("--measure", action="store_true", help="只测量源/目标域高频失配并给出 γ*")
    ap.add_argument("--tasks", type=str, default="TN3K,DDTIz,PTz")
    args = ap.parse_args()

    if args.measure:
        measure(args.ratio)
    else:
        mask = hf_mask(SIZE, SIZE, args.ratio)
        for tag in args.tasks.split(","):
            n = build(tag.strip(), args.gain, args.ratio, mask)
            print(f"gain={args.gain} ratio={args.ratio}: {tag} {n} 张 -> {tag}_hf 就绪")
