#!/usr/bin/env python3
"""Visualize a localization dataset scene file produced by generate_dataset.py.

Renders, for one scene .npz file:
  1. a top-down map: building footprints, the BS (with ULA azimuth), and the
     database / query UE positions;
  2. the CSI magnitude |H| of one sample (antennas x subcarriers);
  3. the CSI phase angle(H) of the same sample.

Saves to a PNG so it works on a headless / SSH machine.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_scene_map(ax, d, area):
    # Buildings
    footprints = d["footprints"]
    for (cx, cy, L, W) in footprints:
        ax.add_patch(Rectangle((cx - L / 2, cy - W / 2), L, W,
                               facecolor="0.7", edgecolor="0.3", zorder=1))
    # UE positions
    ax.scatter(d["pos_db"][:, 0], d["pos_db"][:, 1], s=8, c="tab:blue",
               alpha=0.5, label="database UEs", zorder=2)
    ax.scatter(d["pos_query"][:, 0], d["pos_query"][:, 1], s=24, c="tab:orange",
               marker="x", label="query UEs", zorder=3)
    # BS at origin + ULA azimuth arrow
    az = np.deg2rad(float(d["az_deg"]))
    ax.scatter([0], [0], marker="^", s=160, c="red", zorder=4, label="BS")
    arrow_len = area * 0.25
    ax.annotate("", xy=(arrow_len * np.cos(az), arrow_len * np.sin(az)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="red", lw=2), zorder=4)
    ax.set_xlim(-area, area)
    ax.set_ylim(-area, area)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Scene map  (N_ant={int(d['n_ant'])}, "
                 f"BW={float(d['bw_hz'])/1e6:.0f} MHz, "
                 f"z_bs={float(d['z_bs']):.1f} m, az={float(d['az_deg']):.0f}°)")
    ax.legend(loc="upper right", fontsize=8)


def plot_csi(ax, H, title, cmap, label):
    im = ax.imshow(H, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xlabel("subcarrier index")
    ax.set_ylabel("antenna index")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)


def main():
    ap = argparse.ArgumentParser(description="Visualize a localization scene .npz")
    ap.add_argument("file", help="path to a scene .npz file")
    ap.add_argument("--sample-idx", type=int, default=0,
                    help="which database sample's CSI to show")
    ap.add_argument("--area", type=float, default=50.0, help="map half-extent [m]")
    ap.add_argument("--out", default=None, help="output PNG (default: <file>.png)")
    args = ap.parse_args()

    d = np.load(args.file)
    H = d["H_db"][args.sample_idx]            # (N_ant, N_subc), complex
    ue = d["pos_db"][args.sample_idx]

    fig = plt.figure(figsize=(15, 4.5))
    ax0 = fig.add_subplot(1, 3, 1)
    plot_scene_map(ax0, d, args.area)
    # Highlight the sample whose CSI we display
    ax0.scatter([ue[0]], [ue[1]], s=120, facecolors="none",
                edgecolors="black", linewidths=1.8, zorder=5)

    ax1 = fig.add_subplot(1, 3, 2)
    plot_csi(ax1, np.abs(H), f"|H|  (sample {args.sample_idx} @ "
             f"({ue[0]:.1f}, {ue[1]:.1f}))", "viridis", "|H|")

    ax2 = fig.add_subplot(1, 3, 3)
    plot_csi(ax2, np.angle(H), "angle(H)", "twilight", "phase [rad]")

    fig.tight_layout()
    out = args.out or (args.file + ".png")
    fig.savefig(out, dpi=120)
    print(f"Saved visualization to {out}")


if __name__ == "__main__":
    main()
