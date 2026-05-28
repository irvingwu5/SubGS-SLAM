#!/bin/bash
# 实验1: tracking_init_mode ablation on fr3_office
# 四种模式: prev_only, cv_only, vo_only, multi
# GPU: 1

set -e

CONFIG_DIR="configs/rgbd/tum/ablation_tracking_init"
GPU=1

echo "========== D: prev_only =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/D_prev_only_fr3.yaml --eval

echo "========== E: cv_only =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/E_cv_only_fr3.yaml --eval

echo "========== F: vo_only =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/F_vo_only_fr3.yaml --eval

echo "========== G: multi =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/G_multi_fr3.yaml --eval

echo "========== Done =========="
