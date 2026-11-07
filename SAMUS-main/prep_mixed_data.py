#!/usr/bin/env python3
# =============================================================================
# prep_mixed_data.py — 构造"目标域数据可用"混合训练集(全参微调对照零样本)
#   训练池 = TN3K train(2000) + DDTI(466-47测试=419) + private_Thyroid Train(428-44测试=384)
#   · DDTIz(47)/PTz(44) 是既有零样本跨域测试子集，已确认分别⊂DDTI池/⊂PT Train；
#     必须从训练池剔除，避免 train∩test≠∅ 泄漏。PT Val(527) 无 mask，不参与训练。
#   · 掩码原始即 0/255，与 loader `>=128` 二值化口径一致，直接拷贝不转换(仅防御性 *255)。
#   · 新子集落在 dataset/SAMUS/{DDTI_tr,PT_tr}/{img,label}/*.png；split 用主名(loader 自动试 .jpg/.png)。
#   产物：train_allsrc.txt(MainPatient/, 4字段 class/sub/name/label) + class.json 追加两键。
#   用法：python3 prep_mixed_data.py            (幂等：已存在同名文件则跳过覆盖)
# =============================================================================
import os, json, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DS   = os.path.join(HERE, "dataset", "SAMUS")
MP   = os.path.join(DS, "MainPatient")

# ---- 源目录（相对 KTD/SAMUS-main 的上级 Downloads）----
DL = os.path.abspath(os.path.join(HERE, "..", ".."))          # /home/linux/Downloads
DDTI_IMG  = os.path.join(DL, "challenge_data_public/Challenge_Data_Public/segmentation/DDTI/imgs")
DDTI_MASK = os.path.join(DL, "challenge_data_public/Challenge_Data_Public/segmentation/DDTI/masks")
PTB = os.path.join(DL, "challenge_data_private_v2_fully_anonymized/Challenge_Data_Private_v2_fully_anonymized")
PT_IMG  = os.path.join(PTB, "Train/segmentation/Thyroid/imgs")
PT_MASK = os.path.join(PTB, "Train/segmentation/Thyroid/masks")


def bases(d):
    return {os.path.splitext(f)[0] for f in os.listdir(d)}


def test_bases(sub):
    """既有零样本测试子集的主名（用于剔除）。"""
    p = os.path.join(DS, sub, "img")
    return bases(p) if os.path.isdir(p) else set()


def build_subset(name, img_src, mask_src, exclude):
    """把 img_src\\mask_src 中不在 exclude 的样本拷成 dataset/SAMUS/<name>/{img,label}/*.png。
    返回可用主名列表（排序）。"""
    img_dir  = os.path.join(DS, name, "img")
    label_dir = os.path.join(DS, name, "label")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    import cv2, numpy as np
    kept = []
    for b in sorted(bases(img_src) - exclude):
        si = os.path.join(img_src, b + ".png")
        sm = os.path.join(mask_src, b + ".png")
        if not (os.path.exists(si) and os.path.exists(sm)):
            continue
        di, dm = os.path.join(img_dir, b + ".png"), os.path.join(label_dir, b + ".png")
        if not os.path.exists(di):
            shutil.copyfile(si, di)
        if not os.path.exists(dm):
            m = cv2.imread(sm, 0)
            if m is not None and m.max() <= 1:        # 防御：二值 0/1 → 0/255
                m = (m * 255).astype(np.uint8)
                cv2.imwrite(dm, m)
            else:
                shutil.copyfile(sm, dm)
        kept.append(b)
    return kept


def main():
    # 1) 剔除既有测试子集
    ddti_ex = test_bases("DDTIz")
    pt_ex   = test_bases("PTz")
    print(f"[prep] 剔除测试 DDTIz={len(ddti_ex)}  PTz={len(pt_ex)}")

    ddti_names = build_subset("DDTI_tr", DDTI_IMG, DDTI_MASK, ddti_ex)
    pt_names   = build_subset("PT_tr",   PT_IMG,   PT_MASK,   pt_ex)
    print(f"[prep] DDTI_tr 可用={len(ddti_names)}  PT_tr 可用={len(pt_names)}")

    # 2) class.json 追加两键（沿用 segmentation 类数=2）
    cj = os.path.join(MP, "class.json")
    d = json.load(open(cj))
    d["DDTI_tr"] = 2
    d["PT_tr"] = 2
    json.dump(d, open(cj, "w"), indent=2)
    print(f"[prep] class.json 写入 DDTI_tr/PT_tr，现有键={len(d)}")

    # 3) 组 train_allsrc.txt：TN3K train 原样 + DDTI_tr/PT_tr 追加(4字段 1/sub/name/label)
    tn_lines = [l.strip() for l in open(os.path.join(MP, "train.txt")) if l.strip()]
    lines = list(tn_lines)
    lines += [f"1/DDTI_tr/{b}/1" for b in ddti_names]
    lines += [f"1/PT_tr/{b}/1"   for b in pt_names]
    out = os.path.join(MP, "train_allsrc.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[prep] {out} 写入 {len(lines)} 行 (TN3K {len(tn_lines)} + DDTI_tr {len(ddti_names)} + PT_tr {len(pt_names)})")

    # 4) 自检：随机几行路径可解析
    import random
    random.seed(0)
    for s in random.sample(lines, 6):
        p = s.split("/")
        sub, name = p[1], p[2]
        ip = os.path.join(DS, sub, "img", name)
        ok = any(os.path.exists(ip + e) for e in (".jpg", ".png"))
        lp = os.path.join(DS, sub, "label", name)
        lok = any(os.path.exists(lp + e) for e in (".jpg", ".png"))
        assert ok and lok, f"路径不可解析: {s}"
    print("[prep] 抽样路径自检通过")


if __name__ == "__main__":
    main()
