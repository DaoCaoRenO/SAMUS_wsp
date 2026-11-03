#!/usr/bin/env python3
"""
测试时频谱对齐 (FDA, Yang & Soatto CVPR2020)：
把目标测试集(DDTIz/PTz)每张图的低频幅度谱替换为源域(TN3K train)的平均低频幅度，
保留相位(=内容/位置不变)与 DC(=整体亮度不变)，GT 掩膜原样沿用。
产出 dataset/SAMUS/{tag}_fda/{img,label} + MainPatient/test_{tag}_fda.txt，
之后用现有 test.py --task DDTIz_fda 走完全相同的评测管线，保证口径一致。
"""
import os, json, argparse
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "dataset", "SAMUS")
SIZE = 256


def amp_phase(img_f):
    f = np.fft.fftshift(np.fft.fft2(img_f))
    return np.abs(f), np.angle(f)


def low_freq_mask(H, W, ratio, exclude_dc=True):
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H // 2, W // 2
    r = np.sqrt(((yy - cy) / (H / 2.0)) ** 2 + ((xx - cx) / (W / 2.0)) ** 2)
    m = r < ratio
    if exclude_dc:
        m[cy, cx] = False
    return m


def source_template(names, n_sample):
    """TN3K train 图 -> 平均幅度谱 (256x256)"""
    rng = np.random.RandomState(0)
    picked = rng.choice(len(names), size=min(n_sample, len(names)), replace=False)
    acc = np.zeros((SIZE, SIZE), dtype=np.float64)
    cnt = 0
    for idx in picked:
        sub, name = names[idx].split("/")[1], names[idx].split("/")[2]
        p = os.path.join(DS, sub, "img", name + ".jpg")
        im = cv2.imread(p, 0)
        if im is None:
            continue
        im = cv2.resize(im.astype(np.float32), (SIZE, SIZE))
        amp, _ = amp_phase(im)
        acc += amp
        cnt += 1
    return acc / max(cnt, 1)


def adapt_dir(tag, ratio, tmpl, mask):
    """对 tag 的 test 集做 FDA，写入 {tag}_fda；返回处理张数"""
    manifest = os.path.join(DS, "MainPatient", f"test_{tag}.txt")
    lines = [l.strip() for l in open(manifest) if l.strip()]
    out_root = os.path.join(DS, f"{tag}_fda")
    for d in ("img", "label"):
        os.makedirs(os.path.join(out_root, d), exist_ok=True)
    out_lines = []
    n = 0
    for ln in lines:
        sub, fname, cls = ln.split("/")
        src_img = cv2.imread(os.path.join(DS, sub, "img", fname + ".jpg"), 0)
        if src_img is None:
            continue
        im = cv2.resize(src_img.astype(np.float32), (SIZE, SIZE))
        amp, pha = amp_phase(im)
        amp_new = amp.copy()
        amp_new[mask] = tmpl[mask]  # 低频幅度换成源域统计（DC 保留）
        out = np.real(np.fft.ifft2(np.fft.ifftshift(amp_new * np.exp(1j * pha))))
        out = np.clip(out, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_root, "img", fname + ".jpg"), out)
        # label 直接软链原图
        link = os.path.join(out_root, "label", fname + ".jpg")
        if not os.path.lexists(link):
            os.symlink(os.path.join(DS, sub, "label", fname + ".jpg"), link)
        out_lines.append(f"{tag}_fda/{fname}/{cls}")
        n += 1
    with open(os.path.join(DS, "MainPatient", f"test_{tag}_fda.txt"), "w") as f:
        f.write("\n".join(out_lines) + "\n")
    return n


def register_class(tags):
    p = os.path.join(DS, "MainPatient", "class.json")
    cj = json.load(open(p))
    for t in tags:
        cj[f"{t}_fda"] = 2
    json.dump(cj, open(p, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.1, help="低频半径比例 (0~1)")
    ap.add_argument("--n_sample", type=int, default=300, help="源域统计采样张数")
    args = ap.parse_args()

    train_names = [l.strip() for l in open(os.path.join(DS, "MainPatient", "train.txt")) if l.strip()]
    tmpl = source_template(train_names, args.n_sample)
    mask = low_freq_mask(SIZE, SIZE, args.ratio)

    n1 = adapt_dir("DDTIz", args.ratio, tmpl, mask)
    n2 = adapt_dir("PTz", args.ratio, tmpl, mask)
    register_class(["DDTIz", "PTz"])
    print(f"FDA ratio={args.ratio}: DDTIz {n1} 张, PTz {n2} 张 -> *_fda 就绪")
