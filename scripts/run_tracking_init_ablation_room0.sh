#!/bin/bash
# 实验1: tracking_init_mode ablation on Replica room0
# 四种模式: prev_only, cv_only, vo_only, multi
# GPU: 1

set -e

CONFIG_DIR="configs/rgbd/replica/ablation_tracking_init"
GPU=1

echo "============================================"
echo "  Replica room0  "
echo "============================================"

echo "========== room0 D: prev_only =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/D_prev_only_room0.yaml --eval

echo "========== room0 E: cv_only =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/E_cv_only_room0.yaml --eval

echo "========== room0 F: vo_only =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/F_vo_only_room0.yaml --eval

echo "========== room0 G: multi =========="
CUDA_VISIBLE_DEVICES=${GPU} python slam.py --config ${CONFIG_DIR}/G_multi_room0.yaml --eval

echo "========== Done =========="
