#!/usr/bin/env python
"""Plot camera trajectory overlaid with per-frame PSNR (color) and replay count (radius).

Single mode:
    python scripts/plot_psnr_trajectory.py --save_dir outputs/EXP_NAME

Compare mode (side-by-side):
    python scripts/plot_psnr_trajectory.py \
        --save_dir outputs/EXP_RSKM --label "RSKM (vanilla)" \
        --save_dir outputs/EXP_PAR --label "PAR-RSKM" \
        --compare
"""

import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams.update({"font.size": 11})


def load_data(save_dir):
    psnr_dir = os.path.join(save_dir, "psnr", "global_merged_after_opt")
    psnr = np.loadtxt(os.path.join(psnr_dir, "all_psnr.txt"))
    frame_ids = np.loadtxt(os.path.join(psnr_dir, "all_frame_ids.txt"), dtype=int)
    traj = np.loadtxt(os.path.join(psnr_dir, "trajectory_xy.txt"))
    traj_xy = traj[:, 1:]

    replay_counts = {}
    replay_path = os.path.join(save_dir, "rskm_replay_counts.json")
    if os.path.isfile(replay_path):
        with open(replay_path) as f:
            raw = json.load(f)
            replay_counts = {int(k): v for k, v in raw.items()}
    return psnr, frame_ids, traj_xy, replay_counts


def match_replay_counts(frame_ids, replay_counts):
    counts = np.zeros(len(frame_ids), dtype=int)
    for i, fid in enumerate(frame_ids):
        counts[i] = replay_counts.get(int(fid), 0)
    return counts


def _draw_one(ax, traj_xy, eval_fids, psnr_vals, opt_counts, title, vmin, vmax,
              show_legend=True):
    eval_positions = traj_xy[eval_fids]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sizes = np.maximum(opt_counts / 10.0, 20.0)

    sc = ax.scatter(
        eval_positions[:, 0], eval_positions[:, 1],
        c=psnr_vals, cmap="viridis", norm=norm,
        s=sizes, alpha=0.85, edgecolors="none",
        label="Camera Position (Radius = Opt. Iters)",
    )
    (line,) = ax.plot(
        traj_xy[:, 0], traj_xy[:, 1],
        color="red", linewidth=0.5, label="Camera Trajectory",
    )
    ax.plot(
        traj_xy[0, 0], traj_xy[0, 1],
        marker="o", color="red", markersize=6, label="Start Point",
    )

    psnr_mean = float(np.mean(psnr_vals))
    psnr_var = float(np.var(psnr_vals))
    ax.set_title(
        rf"$\mu_{{\mathrm{{PSNR}}}}$={psnr_mean:.2f} dB, "
        rf"$\sigma_{{\mathrm{{PSNR}}}}^2$={psnr_var:.2f} dB$^2$",
        fontsize=13,
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    if show_legend:
        ax.legend(loc="upper left", fontsize=9, markerscale=0.7,
                  handletextpad=0.5, borderpad=0.3, framealpha=0.85)

    return sc


def plot_single(save_dir, out_name):
    psnr, fids, traj_xy, replay = load_data(save_dir)
    if len(psnr) == 0:
        print("ERROR: No PSNR data found.")
        return

    opt = match_replay_counts(fids, replay)
    if opt.sum() == 0:
        opt = np.ones_like(opt) * 20

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = _draw_one(ax, traj_xy, fids, psnr, opt, "", psnr.min(), psnr.max())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(sc, cax=cax, label="PSNR [dB]")
    fig.tight_layout(pad=0.3)

    out_path = os.path.join(save_dir, out_name) if not os.path.isabs(out_name) else out_name
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved to {out_path}")


def plot_compare(dir_labels, out_path):
    datasets = []
    for save_dir, label in dir_labels:
        psnr, fids, traj_xy, replay = load_data(save_dir)
        opt = match_replay_counts(fids, replay)
        if opt.sum() == 0:
            opt = np.ones_like(opt) * 20
        datasets.append((psnr, fids, traj_xy, opt, label))

    n = len(datasets)
    # Use individual per-subplot colorbar range (not shared), so each
    # subplot's colormap reflects its own PSNR range.
    vmin = min(d[0].min() for d in datasets)
    vmax = max(d[0].max() for d in datasets)

    all_x = np.concatenate([d[2][:, 0] for d in datasets])
    all_y = np.concatenate([d[2][:, 1] for d in datasets])
    x_lim = (all_x.min() - 0.3, all_x.max() + 0.3)
    y_lim = (all_y.min() - 0.3, all_y.max() + 0.3)

    # Subplot letters
    sub_labels = ["(a)", "(b)", "(c)", "(d)"]

    fig, axes = plt.subplots(1, n, figsize=(7.2 * n, 6.5))
    if n == 1:
        axes = [axes]

    for i, (ax, (psnr, fids, traj_xy, opt, label)) in enumerate(zip(axes, datasets)):
        sc = _draw_one(ax, traj_xy, fids, psnr, opt, label, vmin, vmax,
                       show_legend=True)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        # Per-subplot colorbar (exact height match via make_axes_locatable)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("PSNR [dB]", fontsize=11)

        # Subplot caption below x-axis
        ax.set_xlabel(f"X [m]\n{sub_labels[i]} {label}", fontsize=11)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved comparison to {out_path}")
    for psnr, _, _, _, label in datasets:
        s = np.sort(psnr)
        w10 = s[:max(1, int(len(s) * 0.1))].mean()
        print(f"  {label}: μ={np.mean(psnr):.2f}, σ²={np.var(psnr):.2f}, Worst-10%={w10:.2f}, n={len(psnr)}")


def main():
    parser = argparse.ArgumentParser(description="PSNR-trajectory overlay plot")
    parser.add_argument("--save_dir", type=str, action="append", default=[],
                        help="Experiment output dir (repeatable)")
    parser.add_argument("--label", type=str, action="append", default=[],
                        help="Label for each --save_dir (repeatable, same order)")
    parser.add_argument("--out_name", type=str, default="trajectory_psnr.pdf")
    parser.add_argument("--compare", action="store_true",
                        help="Side-by-side in one PDF")
    parser.add_argument("--split", action="store_true",
                        help="Output one PDF per experiment (for LaTeX subfigure)")
    args = parser.parse_args()

    if len(args.save_dir) == 0:
        print("ERROR: at least one --save_dir required.")
        return

    # Pad labels if fewer given
    labels = args.label + [f"Exp {i}" for i in range(len(args.label), len(args.save_dir))]

    if args.compare and args.split:
        print("ERROR: --compare and --split are mutually exclusive.")
        return

    if args.compare:
        out_path = os.path.join(os.path.commonpath(args.save_dir), args.out_name) \
            if os.path.commonpath(args.save_dir) else args.out_name
        plot_compare(list(zip(args.save_dir, labels)), out_path)
    elif args.split:
        for (save_dir, label) in zip(args.save_dir, labels):
            out_name = f"trajectory_psnr_{label.replace(' ', '_').replace('(', '').replace(')', '')}.pdf"
            plot_single(save_dir, out_name)
    else:
        plot_single(args.save_dir[0], args.out_name)


if __name__ == "__main__":
    main()
