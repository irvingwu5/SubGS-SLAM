#!/bin/sh

# TUM fr3_office — RSKM / PAR-RSKM 消融实验 (GPU 1)
#
#   A: 2DGS Base (LCKM)           — no VO, no RSKM
#   B: 2DGS Base + RSKM (vanilla) — no VO, vanilla RSKM
#   C: 2DGS Base + VO + RSKM      — VO direct init + vanilla RSKM
#   D: Full VPAR-GS-SLAM          — VO direct init + PAR-RSKM
#
#   All share SA depth + SA dist, no FFT/ErrorMask, no submap/loop.

export CUDA_VISIBLE_DEVICES=1

echo "===== A: 2DGS Base (LCKM) ====="
python slam.py --config configs/rgbd/tum/ablation_rskm/A_lckm_base.yaml --eval 2>&1 | tee outputs/aba_rskm_A_lckm_base.log

echo "===== B: 2DGS Base + RSKM (vanilla) ====="
python slam.py --config configs/rgbd/tum/ablation_rskm/B_rskm_vanilla.yaml --eval 2>&1 | tee outputs/aba_rskm_B_rskm_vanilla.log

echo "===== C: 2DGS Base + VO + RSKM ====="
python slam.py --config configs/rgbd/tum/ablation_rskm/C_vo_rskm_vanilla.yaml --eval 2>&1 | tee outputs/aba_rskm_C_vo_rskm_vanilla.log

echo "===== D: Full VPAR-GS-SLAM (VO + PAR-RSKM) ====="
python slam.py --config configs/rgbd/tum/ablation_rskm/D_vo_par_rskm.yaml --eval 2>&1 | tee outputs/aba_rskm_D_vo_par_rskm.log

echo "===== Ablation complete ====="
echo "Visualization: python scripts/plot_psnr_trajectory.py --save_dir <exp_output_dir>"
