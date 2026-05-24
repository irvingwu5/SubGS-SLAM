#!/usr/bin/env python
"""Regenerate trajectory PSNR data from an old experiment that predates the auto-save feature.

Reads mesh_rendering/{color_*.png, render_poses.json} + GT dataset → computes PSNR,
saves all_psnr.txt, all_frame_ids.txt, trajectory_xy.txt into the experiment's
psnr/global_merged_after_opt/ directory.

Usage:
    python scripts/regenerate_trajectory_data.py \
        --exp_dir /opt/results/Ours/TUM_RGBD/Final/2026-05-09-11-46-22_fftvo_params_opt/
"""

import argparse
import json
import os

import cv2
import numpy as np


def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def load_dataset_frame(dataset_path, frame_id):
    """Load GT RGB image from TUM dataset for a given frame_id.
    TUM dataset: rgb.txt has timestamps + filenames, depth.txt similar.
    We match by sorted line index = frame_id.
    """
    rgb_txt = os.path.join(dataset_path, "rgb.txt")
    with open(rgb_txt) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    # lines: timestamp filepath
    if frame_id >= len(lines):
        raise IndexError(f"frame_id {frame_id} >= {len(lines)}")
    rgb_path = os.path.join(dataset_path, lines[frame_id].split()[-1])
    img = cv2.imread(rgb_path)
    if img is None:
        raise FileNotFoundError(rgb_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Override dataset path (auto-detected from config.yml if omitted)")
    args = parser.parse_args()

    # Resolve dataset path
    dataset_path = args.dataset_path
    if dataset_path is None:
        import yaml
        config_path = os.path.join(args.exp_dir, "config.yml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        dataset_path = cfg["Dataset"]["dataset_path"]
    print(f"Dataset: {dataset_path}")

    # Load render poses
    poses_path = os.path.join(args.exp_dir, "mesh_rendering", "render_poses.json")
    with open(poses_path) as f:
        render_poses = json.load(f)
    # Keys are string frame IDs, values are 4x4 W2C matrices
    fids = sorted(int(k) for k in render_poses.keys())
    print(f"Found {len(fids)} rendered frames (IDs {fids[0]}..{fids[-1]})")

    # Compute PSNR per frame
    psnr_list = []
    positions = []
    for fid in fids:
        color_path = os.path.join(args.exp_dir, "mesh_rendering", f"color_{fid:05d}.png")
        if not os.path.isfile(color_path):
            print(f"  WARNING: missing {color_path}, skipping")
            continue
        rendered = cv2.imread(color_path)
        rendered_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)

        gt_rgb = load_dataset_frame(dataset_path, fid)

        p = psnr(rendered_rgb, gt_rgb)
        psnr_list.append(p)

        w2c = np.array(render_poses[str(fid)])
        positions.append([fid, w2c[0, 3], w2c[1, 3]])

    # Save
    out_dir = os.path.join(args.exp_dir, "psnr", "global_merged_after_opt")
    os.makedirs(out_dir, exist_ok=True)

    np.savetxt(os.path.join(out_dir, "all_psnr.txt"), np.array(psnr_list), fmt="%.6f")
    np.savetxt(os.path.join(out_dir, "all_frame_ids.txt"), np.array(fids), fmt="%d")
    np.savetxt(os.path.join(out_dir, "trajectory_xy.txt"), np.array(positions),
               fmt=["%d", "%.6f", "%.6f"])

    print(f"Regenerated {len(psnr_list)} frames → {out_dir}")
    print(f"  μ_PSNR: {np.mean(psnr_list):.2f}, σ²_PSNR: {np.var(psnr_list):.2f}")


if __name__ == "__main__":
    main()
