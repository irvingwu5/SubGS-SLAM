#!/bin/bash
# 实验1: tracking_init_mode ablation on fr1_desk + fr2_xyz
# 四种模式: prev_only, cv_only, vo_only, multi
# GPU: 1

set -e

CONFIG_DIR="configs/rgbd/tum/ablation_tracking_init"
GPU=0

echo "============================================"
echo "  fr1_desk  "
echo "============================================"

echo "========== fr1_desk D: prev_only =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/D_prev_only_fr1.yaml --eval

echo "========== fr1_desk E: cv_only =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/E_cv_only_fr1.yaml --eval

echo "========== fr1_desk F: vo_only =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/F_vo_only_fr1.yaml --eval

echo "========== fr1_desk G: multi =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/G_multi_fr1.yaml --eval

echo "============================================"
echo "  fr2_xyz  "
echo "============================================"

echo "========== fr2_xyz D: prev_only =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/D_prev_only_fr2.yaml --eval

echo "========== fr2_xyz E: cv_only =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/E_cv_only_fr2.yaml --eval

echo "========== fr2_xyz F: vo_only =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/F_vo_only_fr2.yaml --eval

echo "========== fr2_xyz G: multi =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/G_multi_fr2.yaml --eval

echo "========== Done =========="
