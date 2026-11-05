#!/usr/bin/env python3
"""
prepare_usic_data.py — 将 USIC 多器官数据（无甲状腺）适配到 SAMUS-main 格式。
  1. 为每个非甲状腺数据集创建符号链接目录 dataset/SAMUS/USIC_{name}/{img|label}
  2. 生成统一 train_multiorgan.txt (USIC all organs + TN3K train)
  3. 更新 class.json 加入 USIC_* 键

用法：python3 prepare_usic_data.py [--dry_run]
"""
import os, sys, json, glob
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(HERE, "dataset", "SAMUS")
MAINPATIENT = os.path.join(DATASET_ROOT, "MainPatient")

# Source paths
PUBLIC_SEG = "/home/linux/Downloads/challenge_data_public/Challenge_Data_Public/segmentation"
PRIVATE_SEG = "/home/linux/Downloads/challenge_data_private_v2_fully_anonymized/Challenge_Data_Private_v2_fully_anonymized/Train/segmentation"

# Non-thyroid public datasets: (source_dir_name, target_sub_name)
PUBLIC_DATASETS = [
    ("BUS-BRA",  "USIC_BUSBRA"),
    ("BUSI",     "USIC_BUSI"),
    ("BUSIS",    "USIC_BUSIS"),
    ("CAMUS",    "USIC_CAMUS"),
    ("Fetal_HC", "USIC_FetalHC"),
    ("KidneyUS", "USIC_KidneyUS"),
]
# Non-thyroid private datasets (exclude Thyroid)
PRIVATE_DATASETS = [
    ("Breast",         "USIC_pBreast"),
    ("Breast_luminal", "USIC_pBreastLum"),
    ("Cardiac",        "USIC_pCardiac"),
    ("Fetal_Head",     "USIC_pFetalHead"),
    ("Kidney",         "USIC_pKidney"),
]


def make_symlinks(src_root, datasets, dry=False):
    """Create dataset/SAMUS/{name}/img → src_root/{ds}/imgs, label → masks"""
    total = 0
    for ds_name, target_name in datasets:
        src = os.path.join(src_root, ds_name)
        if not os.path.isdir(src):
            print(f"  [SKIP] {src} not found")
            continue
        img_src = os.path.join(src, "imgs")
        mask_src = os.path.join(src, "masks")
        tgt_dir = os.path.join(DATASET_ROOT, target_name)
        tgt_img = os.path.join(tgt_dir, "img")
        tgt_lbl = os.path.join(tgt_dir, "label")
        if not dry:
            os.makedirs(tgt_dir, exist_ok=True)
            # symlink directories (remove existing if any)
            for link, source in [(tgt_img, img_src), (tgt_lbl, mask_src)]:
                if os.path.islink(link):
                    os.remove(link)
                elif os.path.isdir(link):
                    pass  # keep real dirs if they exist
                if not os.path.exists(link):
                    os.symlink(source, link)
        n = len(os.listdir(img_src)) if os.path.isdir(img_src) else 0
        print(f"  {target_name:20s} ← {ds_name:15s} ({n} imgs)")
        total += n
    return total


def generate_split_entry(target_name, img_dir):
    """Generate train.txt lines for one dataset: '1/{target_name}/{stem}/0'"""
    lines = []
    if not os.path.isdir(img_dir):
        return lines
    for f in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(f)[0]
        lines.append(f"1/{target_name}/{stem}/0")
    return lines


def main():
    dry = "--dry_run" in sys.argv
    print(f"[prepare] === Experiment A: USIC multi-organ + TN3K (NO thyroid) ===")
    print(f"[prepare] dry_run={dry}\n")

    # Step 1: symlinks
    print("[1] Creating symlinks...")
    n_pub = make_symlinks(PUBLIC_SEG, PUBLIC_DATASETS, dry)
    n_pri = make_symlinks(PRIVATE_SEG, PRIVATE_DATASETS, dry)
    print(f"  Public non-thyroid: {n_pub} imgs")
    print(f"  Private non-thyroid: {n_pri} imgs")

    # Step 2: generate train_multiorgan.txt
    print("\n[2] Generating train_multiorgan.txt...")
    all_lines = []
    for _, target_name in PUBLIC_DATASETS + PRIVATE_DATASETS:
        img_dir = os.path.join(DATASET_ROOT, target_name, "img")
        if os.path.isdir(img_dir):
            entries = generate_split_entry(target_name, img_dir)
            all_lines.extend(entries)
            print(f"  {target_name}: {len(entries)} entries")
        else:
            print(f"  {target_name}: [NO DIR] skipping")

    # Append TN3K train entries (from existing train.txt)
    tn3k_train = os.path.join(MAINPATIENT, "train.txt")
    if os.path.isfile(tn3k_train):
        with open(tn3k_train) as f:
            tn3k_lines = [l.strip() for l in f if l.strip()]
        all_lines.extend(tn3k_lines)
        print(f"  TN3K train: {len(tn3k_lines)} entries")

    out_split = os.path.join(MAINPATIENT, "train_multiorgan.txt")
    if not dry:
        with open(out_split, "w") as f:
            f.write("\n".join(all_lines) + "\n")
    print(f"\n  TOTAL training images: {len(all_lines)}")
    print(f"  Written to: {out_split}")

    # Step 3: update class.json
    print("\n[3] Updating class.json...")
    class_file = os.path.join(MAINPATIENT, "class.json")
    with open(class_file) as f:
        cd = json.load(f)
    for _, target_name in PUBLIC_DATASETS + PRIVATE_DATASETS:
        cd[target_name] = 2
    if not dry:
        with open(class_file, "w") as f:
            json.dump(cd, f)
    print(f"  Added {len(PUBLIC_DATASETS)+len(PRIVATE_DATASETS)} keys → class.json")
    print(f"  Keys now: {list(cd.keys())}")

    # Step 4: verify a random sample can be loaded
    print("\n[4] Quick verification (sample 3 entries)...")
    import cv2, numpy as np
    samples = []
    for _, tn in PUBLIC_DATASETS[:2] + PRIVATE_DATASETS[:1]:
        img_dir = os.path.join(DATASET_ROOT, tn, "img")
        if os.path.isdir(img_dir):
            files = sorted(os.listdir(img_dir))
            if files:
                f0 = files[0]
                stem = os.path.splitext(f0)[0]
                img_path = os.path.join(img_dir, f0)
                lbl_path = os.path.join(DATASET_ROOT, tn, "label", f0)
                img = cv2.imread(img_path, 0)
                mask = cv2.imread(lbl_path, 0)
                ok = img is not None and mask is not None
                print(f"  {tn}/{stem}: img={img.shape if ok else 'FAIL'} mask={mask.shape if ok else 'FAIL'} {'OK' if ok else 'FAIL'}")

    print("\n[prepare] DONE. Now run:")
    print(f"  python train.py --modelname SAMUS --task MultiOrgan --epochs 100 --out_tag multiorgan")


if __name__ == "__main__":
    main()
