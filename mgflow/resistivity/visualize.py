#!/usr/bin/env python3
"""Generate publication-quality figures for resistivity analysis.

Usage:
    python visualize.py
"""

import sys, os, json
from pathlib import Path
import numpy as np

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm, colors
from matplotlib.tri import Triangulation, TriAnalyzer

_HERE = Path(__file__).resolve().parent
PRED_DIR = _HERE / "predictions"
FIG_DIR = _HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

ELEMENTS = ["Ta", "Nb", "Ni", "Co", "Hf"]


def barycentric_to_xy(a, b, c):
    x = 0.5 * (2 * b + c) / (a + b + c)
    y = (np.sqrt(3) / 2) * c / (a + b + c)
    return x, y


# ══════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════

pred_path = PRED_DIR / "TaNbNiCoHf_resistivity_predictions.json"
if not pred_path.exists():
    print(f"ERROR: {pred_path} not found. Run training+grid search first.")
    sys.exit(1)

with open(pred_path) as f:
    data = json.load(f)

R = np.array([d["R (µΩ·cm)"] for d in data])
fracs = {el: np.array([d["composition"].get(el, 0.0) for d in data]) for el in ELEMENTS}
frac_matrix = np.column_stack([fracs[el] for el in ELEMENTS])
norm_r = plt.Normalize(R.min(), R.max())
print(f"Loaded {len(data)} predictions, R ∈ [{R.min():.2f}, {R.max():.2f}]")


# ══════════════════════════════════════════════════════
# FIG 1: Overview (6 subplots)
# ══════════════════════════════════════════════════════

def fig_overview():
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.5])

    # (a) Top 15 lowest
    ax1 = fig.add_subplot(gs[0, 0])
    idx_sort = np.argsort(R)
    inds = idx_sort[:15]
    labels_a = []
    for i in inds:
        comp = data[i]["composition"]
        labels_a.append("\n".join(f"{el}{v*100:.0f}" for el, v in
                                  sorted(comp.items(), key=lambda x: -x[1]) if v >= 0.05))
    colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 15))[::-1]
    ax1.barh(range(15), R[inds], color=colors_a, edgecolor="k", lw=0.3)
    ax1.set_yticks(range(15)); ax1.set_yticklabels(labels_a, fontsize=6)
    ax1.set_xlabel("R (µΩ·cm)", fontsize=9)
    ax1.set_title("(a) Lowest Resistivity", fontsize=10, fontweight="bold")
    ax1.invert_yaxis(); ax1.grid(True, alpha=0.2, axis="x")

    # (b) Top 15 highest
    ax2 = fig.add_subplot(gs[0, 1])
    inds2 = idx_sort[-15:][::-1]
    labels_b = []
    for i in inds2:
        comp = data[i]["composition"]
        labels_b.append("\n".join(f"{el}{v*100:.0f}" for el, v in
                                  sorted(comp.items(), key=lambda x: -x[1]) if v >= 0.05))
    colors_b = plt.cm.Reds(np.linspace(0.3, 0.9, 15))
    ax2.barh(range(15), R[inds2], color=colors_b, edgecolor="k", lw=0.3)
    ax2.set_yticks(range(15)); ax2.set_yticklabels(labels_b, fontsize=6)
    ax2.set_xlabel("R (µΩ·cm)", fontsize=9)
    ax2.set_title("(b) Highest Resistivity", fontsize=10, fontweight="bold")
    ax2.invert_yaxis(); ax2.grid(True, alpha=0.2, axis="x")

    # (c) Parallel coordinates
    ax3 = fig.add_subplot(gs[0, 2])
    n_show = min(500, len(R))
    idx_pc = np.random.RandomState(42).choice(len(R), n_show, replace=False)
    cmap_pc = plt.cm.plasma_r
    for idx in idx_pc:
        ax3.plot(range(5), frac_matrix[idx], color=cmap_pc(norm_r(R[idx])),
                 lw=0.3, alpha=0.5)
    ax3.set_xticks(range(5)); ax3.set_xticklabels(ELEMENTS, fontsize=9)
    ax3.set_ylabel("Atomic Fraction", fontsize=9); ax3.set_ylim(-0.02, 1.02)
    ax3.set_title("(c) Parallel Coordinates (colored by R)", fontsize=10, fontweight="bold")
    ax3.grid(True, alpha=0.15)
    sm = plt.cm.ScalarMappable(cmap=cmap_pc, norm=norm_r); sm.set_array([])
    plt.colorbar(sm, ax=ax3, shrink=0.8, pad=0.02).set_label("R (µΩ·cm)", fontsize=8)

    # (d) Nb fraction vs R
    ax4 = fig.add_subplot(gs[1, 0])
    sc4 = ax4.scatter(fracs["Nb"], R, c=fracs["Ni"], s=3, cmap="coolwarm",
                      alpha=0.3, edgecolors="none")
    ax4.set_xlabel("Nb fraction", fontsize=9); ax4.set_ylabel("R (µΩ·cm)", fontsize=9)
    ax4.set_title("(d) R vs Nb fraction (color=Ni)", fontsize=10, fontweight="bold")
    ax4.grid(True, alpha=0.15)
    plt.colorbar(sc4, ax=ax4, shrink=0.8).set_label("Ni fraction", fontsize=8)

    # (e) Number of elements vs R
    ax5 = fig.add_subplot(gs[1, 1])
    n_elem = (frac_matrix > 0.01).sum(axis=1)
    ax5.scatter(n_elem + np.random.uniform(-0.15, 0.15, len(n_elem)), R,
                c=fracs["Nb"], s=2, cmap="viridis", alpha=0.3, edgecolors="none")
    ax5.set_xlabel("Number of elements", fontsize=9); ax5.set_ylabel("R (µΩ·cm)", fontsize=9)
    ax5.set_title("(e) R vs Composition Complexity", fontsize=10, fontweight="bold")
    ax5.set_xticks([1, 2, 3, 4, 5]); ax5.grid(True, alpha=0.15)

    # (f) Hf+Ta vs R
    ax6 = fig.add_subplot(gs[1, 2])
    x_ht = fracs["Hf"] + fracs["Ta"]
    sc6 = ax6.scatter(x_ht, R, c=fracs["Co"], s=3, cmap="plasma", alpha=0.3, edgecolors="none")
    ax6.set_xlabel("Hf + Ta fraction", fontsize=9); ax6.set_ylabel("R (µΩ·cm)", fontsize=9)
    ax6.set_title("(f) R vs (Hf+Ta) fraction (color=Co)", fontsize=10, fontweight="bold")
    ax6.grid(True, alpha=0.15)
    plt.colorbar(sc6, ax=ax6, shrink=0.8).set_label("Co fraction", fontsize=8)

    path = FIG_DIR / "R_overview.png"
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════
# FIG 2: Ternary diagrams (fixed Ta:Nb ratios)
# ══════════════════════════════════════════════════════

def fig_ternary():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.15, left=0.03, right=0.87, top=0.93, bottom=0.02)
    vmin, vmax = 94, 125

    ratios = [
        ("Ta:Nb = 3:1", 0.75, 0.25), ("Ta:Nb = 1:1", 0.50, 0.50),
        ("Ta:Nb = 1:3", 0.25, 0.75), ("Ta:Nb = 4:1", 0.80, 0.20),
        ("Ta:Nb = 1:4", 0.20, 0.80), ("Ta:Nb = 2:3", 0.40, 0.60),
    ]

    total_tn = fracs["Ta"] + fracs["Nb"]

    for idx, (label, ta_r, nb_r) in enumerate(ratios):
        ax = axes[idx // 3, idx % 3]
        target = ta_r / (ta_r + nb_r)
        mask = (np.abs(fracs["Ta"] / (total_tn + 1e-12) - target) < 0.08) & (total_tn > 0.05) & (total_tn < 0.95)

        if mask.sum() < 5:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.text(0.85, 0.06, label,
                    transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.75))
            ax.set_title("")
            continue

        total_nch = fracs["Ni"][mask] + fracs["Co"][mask] + fracs["Hf"][mask]
        a, b, c = [fracs[el][mask] / total_nch for el in ("Ni", "Co", "Hf")]
        x_pts, y_pts = barycentric_to_xy(a, b, c)
        r_vals = R[mask]

        if len(x_pts) >= 10:
            tri = Triangulation(x_pts, y_pts)
            tri.set_mask(TriAnalyzer(tri).get_flat_tri_mask(0.03))
            ax.tripcolor(tri, r_vals, shading="gouraud", cmap="plasma",
                         vmin=vmin, vmax=vmax)
            ax.tricontour(tri, r_vals, levels=8, colors="k", linewidths=0.4,
                          alpha=0.3, vmin=vmin, vmax=vmax)

        tri_verts = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
        ax.plot(tri_verts[:, 0], tri_verts[:, 1], "k-", lw=0.6)
        for pos, lbl in [((0, -0.03), "Ni"), ((1, -0.03), "Co"),
                         ((0.5, np.sqrt(3)/2+0.025), "Hf")]:
            ax.text(*pos, lbl, ha="center", va="top", fontsize=7, fontweight="bold")
        ax.set_aspect("equal"); ax.axis("off")
        # Put ratio label inside the triangle (bottom-right), title is just letter
        ax.text(0.85, 0.06, label,
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.75))
        ax.set_title("")

    cbar_ax = fig.add_axes([0.88, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin, vmax)); sm.set_array([])
    plt.colorbar(sm, cax=cbar_ax).set_label("R (µΩ·cm)", fontsize=10)
    fig.suptitle("Ni-Co-Hf Ternary at Fixed Ta:Nb Ratios", fontsize=13, fontweight="bold", y=0.98)

    path = FIG_DIR / "R_ternary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════
# FIG 3: Scatter matrix
# ══════════════════════════════════════════════════════

def fig_scatter_matrix():
    fig = plt.figure(figsize=(18, 15), constrained_layout=True)
    gs = gridspec.GridSpec(5, 5, figure=fig)

    for i in range(5):
        for j in range(5):
            if i == j:
                ax = fig.add_subplot(gs[i, j])
                ax.hist(fracs[ELEMENTS[i]], bins=30, color="steelblue",
                        alpha=0.7, edgecolor="k", lw=0.3)
                ax.set_xlim(0, 1); ax.set_title(ELEMENTS[i], fontsize=10, fontweight="bold")
                if i < 4: ax.tick_params(labelbottom=False)
                if j > 0: ax.tick_params(labelleft=False)
            elif i < j:
                ax = fig.add_subplot(gs[i, j])
                ax.scatter(fracs[ELEMENTS[j]], fracs[ELEMENTS[i]],
                           c=R, cmap="plasma", s=2, alpha=0.35, norm=norm_r, edgecolors="none")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                if i < 4: ax.tick_params(labelbottom=False)
                if j > 0: ax.tick_params(labelleft=False)
                if i == 0: ax.set_title(ELEMENTS[j], fontsize=9)
            else:
                fig.add_subplot(gs[i, j]).axis("off")

    cbar_ax = fig.add_axes([0.92, 0.10, 0.012, 0.80])
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm_r); sm.set_array([])
    plt.colorbar(sm, cax=cbar_ax).set_label("R (µΩ·cm)", fontsize=11)
    fig.suptitle("Pairwise Fraction Correlations (color = R)", fontsize=14, fontweight="bold", y=0.98)

    path = FIG_DIR / "R_scatter_matrix.png"
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════
# FIG 4: 3D scatter
# ══════════════════════════════════════════════════════

def fig_3d():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    sub = np.random.RandomState(42).choice(len(R), min(1500, len(R)), replace=False)
    size = (fracs["Co"][sub] + fracs["Hf"][sub] + 0.01) * 80
    colors_3d = plt.cm.plasma(norm_r(R[sub]))

    ax.scatter(fracs["Nb"][sub], fracs["Ta"][sub], fracs["Ni"][sub],
               c=colors_3d, s=size, alpha=0.6, edgecolors="none")
    ax.set_xlabel("Nb fraction", fontsize=10, labelpad=8)
    ax.set_ylabel("Ta fraction", fontsize=10, labelpad=8)
    ax.set_zlabel("Ni fraction", fontsize=10, labelpad=6)
    ax.set_title("3D Composition Space (color=R, size=Co+Hf)", fontsize=12, fontweight="bold")

    cbar_ax = fig.add_axes([0.88, 0.12, 0.025, 0.76])
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm_r); sm.set_array([])
    plt.colorbar(sm, cax=cbar_ax).set_label("R (µΩ·cm)", fontsize=11)

    path = FIG_DIR / "R_3d_scatter.png"
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════
# FIG 5: Trend lines
# ══════════════════════════════════════════════════════

def fig_trends():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    desc_list = [
        ("Nb fraction", fracs["Nb"]), ("Ta fraction", fracs["Ta"]),
        ("Ni fraction", fracs["Ni"]), ("Co fraction", fracs["Co"]),
        ("Hf fraction", fracs["Hf"]),
        ("Number of elements", (frac_matrix > 0.01).sum(axis=1).astype(float)),
    ]

    for idx_ax, (label, x_vals) in enumerate(desc_list):
        ax = axes.ravel()[idx_ax]
        ax.scatter(x_vals, R, c=R, cmap="plasma", s=3, alpha=0.2, norm=norm_r, edgecolors="none")
        ax.grid(True, alpha=0.15); ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("R (µΩ·cm)", fontsize=10)

        if idx_ax < 5:
            bins = np.linspace(x_vals.min(), x_vals.max(), 15)
            bc = (bins[:-1] + bins[1:]) / 2
            bm = np.array([R[(x_vals >= bins[i]) & (x_vals < bins[i+1])].mean()
                           for i in range(len(bins)-1)])
            bs = np.array([R[(x_vals >= bins[i]) & (x_vals < bins[i+1])].std()
                           for i in range(len(bins)-1)])
            valid = ~np.isnan(bm)
            ax.plot(bc[valid], bm[valid], "r-", lw=2, alpha=0.8)
            ax.fill_between(bc[valid], bm[valid]-bs[valid], bm[valid]+bs[valid],
                            color="r", alpha=0.08)

    fig.suptitle("Resistivity vs Composition Descriptors (red = mean ± std)",
                 fontsize=13, fontweight="bold", y=1.01)
    path = FIG_DIR / "R_trends.png"
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures...")
    fig_overview()
    fig_ternary()
    fig_scatter_matrix()
    fig_3d()
    fig_trends()
    print(f"All figures saved to {FIG_DIR}/")
