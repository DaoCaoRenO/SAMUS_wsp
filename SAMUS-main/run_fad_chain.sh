#!/usr/bin/env bash
# 方案C 链式训练：先 C 单独(fad50)，再 A+C(fadaug50)。均为 50 轮，与 base50/freq50 严格同条件。
# 用法：setsid nohup bash run_fad_chain.sh > /dev/null 2>&1 &
set -u
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
cd /home/linux/Downloads/KTD/SAMUS-main || exit 1

CHAIN=train_chain2.log

echo "[START fad50] $(date)" >> "$CHAIN"
python train.py --modelname SAMUS --task TN3K --epochs 50 --out_tag fad50 \
       --freq_aug 0 --freq_adapter 1 > train_fad50.log 2>&1
echo "[DONE fad50] $(date) rc=$?" >> "$CHAIN"

echo "[START fadaug50] $(date)" >> "$CHAIN"
python train.py --modelname SAMUS --task TN3K --epochs 50 --out_tag fadaug50 \
       --freq_aug 1 --freq_adapter 1 > train_fadaug50.log 2>&1
echo "[DONE fadaug50] $(date) rc=$?" >> "$CHAIN"

echo "[ALL DONE] $(date)" >> "$CHAIN"
