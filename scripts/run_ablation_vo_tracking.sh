#!/bin/sh

# TUM fr3_office — VO-Prior-Guided Tracking 消融实验 (GPU 0)
#
#   A: Render-only Tracking         (无 VO，纯渲染精化 100 iters)
#   B: VO Init. + Gaussian Refine   (VO 初值 + 候选选择 + 渲染精化 60 iters)
#   C: VO + Gate + Gaussian Refine  (VO 初值 + 候选选择 + 渲染 gate + 精化)
#
#   All use the same 2DGS mapping baseline (no FFT/ErrorMask/RSKM/SA).

export CUDA_VISIBLE_DEVICES=1

#echo "===== A: Render-only Tracking ====="
#python slam.py --config configs/rgbd/tum/ablation_vo/A_render_only.yaml --eval 2>&1 | tee outputs/aba_vo_tracking_A_render_only.log

echo "===== B: VO Init. + Gaussian Refinement ====="
python slam.py --config configs/rgbd/tum/ablation_vo/B_vo_init_refine.yaml --eval 2>&1 | tee outputs/aba_vo_tracking_B_vo_init_refine.log

echo "===== C: VO + Gate + Gaussian Refinement ====="
python slam.py --config configs/rgbd/tum/ablation_vo/C_vo_gate_refine.yaml --eval 2>&1 | tee outputs/aba_vo_tracking_C_vo_gate_refine.log

echo "===== Ablation complete ====="
echo "Track Time [ms] and Iteration Count printed in each log above."
