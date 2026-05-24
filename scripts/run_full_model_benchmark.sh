#!/bin/sh

# Full Model Benchmark: 2DGS Base + SA depth/dist + VO + PAR-RSKM
# GPU 1

export CUDA_VISIBLE_DEVICES=1

# ============================================================
# Replica
# ============================================================
for scene in office0 office1 office2 office3 office4 room0 room1 room2; do
    echo "===== Replica ${scene} ====="
    python slam.py --config configs/rgbd/replica/full_model/${scene}.yaml --eval 2>&1 | tee outputs/full_replica_${scene}.log
done

# ============================================================
# TUM
# ============================================================
for scene in fr1_desk fr2_xyz; do
    echo "===== TUM ${scene} ====="
    python slam.py --config configs/rgbd/tum/full_model/${scene}.yaml --eval 2>&1 | tee outputs/full_tum_${scene}.log
done

echo "===== Full Model Benchmark Complete ====="
