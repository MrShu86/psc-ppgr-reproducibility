from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "final"
OUT_BASE = OUT_DIR / "Fig2_PSC_PPGR_architecture"


COLORS = {
    "data": "#D9EAF7",
    "model": "#E8E1F5",
    "psc": "#D8F0F2",
    "output": "#ECEFF3",
    "accent": "#2C7FB8",
    "text": "#1F2D3D",
    "muted": "#566573",
    "border": "#34495E",
    "white": "#FFFFFF",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
        }
    )


def box(ax, xy, wh, label, fc, ec=None, lw=1.1, fs=7.0, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        facecolor=fc,
        edgecolor=ec or COLORS["border"],
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight=weight,
        color=COLORS["text"],
        linespacing=1.12,
    )
    return patch


def arrow(ax, start, end, color=None, lw=1.25, rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=lw,
        color=color or COLORS["border"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(patch)
    return patch


def header(ax, x, y, label, color):
    ax.text(
        x,
        y,
        label,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.plot([x, x + 0.105], [y - 0.010, y - 0.010], color=color, lw=2.0, solid_capstyle="round")


def region(ax, x, y, w, h, color):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.020",
        facecolor=color,
        edgecolor=color,
        linewidth=0.9,
        alpha=0.24,
    )
    ax.add_patch(patch)


def draw_input(ax):
    region(ax, 0.035, 0.18, 0.215, 0.68, COLORS["data"])
    header(ax, 0.052, 0.815, "Input meal events", COLORS["accent"])

    box(ax, (0.062, 0.610), (0.082, 0.085), "Query\nmeal", COLORS["data"], fs=7.2, weight="bold")
    box(ax, (0.062, 0.355), (0.082, 0.105), "K-shot\nsupport", COLORS["data"], fs=7.2, weight="bold")

    feature_x = 0.165
    features = [
        ("Pre-meal\nCGM", 0.655),
        ("Nutrition\nvector", 0.555),
        ("Device/domain", 0.455),
        ("Subject/context", 0.355),
    ]
    for text, y in features:
        box(ax, (feature_x, y), (0.070, 0.062), text, COLORS["white"], ec=COLORS["accent"], lw=1.0, fs=6.5)

    arrow(ax, (0.144, 0.653), (feature_x, 0.686), COLORS["accent"], 1.05)
    arrow(ax, (0.144, 0.641), (feature_x, 0.586), COLORS["accent"], 1.05)
    arrow(ax, (0.144, 0.408), (feature_x, 0.486), COLORS["accent"], 1.05, rad=0.12)
    arrow(ax, (0.144, 0.390), (feature_x, 0.386), COLORS["accent"], 1.05)


def draw_model(ax):
    region(ax, 0.280, 0.18, 0.245, 0.68, COLORS["model"])
    header(ax, 0.295, 0.815, "Base event predictor", COLORS["model"])

    box(ax, (0.302, 0.625), (0.082, 0.075), "Tabular\nbranch", COLORS["model"], fs=6.8)
    box(ax, (0.302, 0.425), (0.082, 0.075), "Temporal CGM\nbranch", COLORS["model"], fs=6.8)
    box(ax, (0.420, 0.520), (0.078, 0.090), "Event\nrepresentation", COLORS["model"], fs=6.8, weight="bold")

    arrow(ax, (0.235, 0.686), (0.302, 0.666), lw=1.25)
    arrow(ax, (0.235, 0.586), (0.302, 0.466), lw=1.25, rad=-0.10)
    arrow(ax, (0.384, 0.662), (0.420, 0.573), lw=1.15)
    arrow(ax, (0.384, 0.462), (0.420, 0.557), lw=1.15)

    box(ax, (0.420, 0.350), (0.078, 0.075), "Initial PPGR", COLORS["white"], ec=COLORS["border"], fs=6.8)
    arrow(ax, (0.459, 0.520), (0.459, 0.425), lw=1.15)


def draw_psc(ax):
    region(ax, 0.555, 0.18, 0.245, 0.68, COLORS["psc"])
    header(ax, 0.570, 0.805, "Personalized\nSupport Calibration", COLORS["psc"])

    box(ax, (0.575, 0.635), (0.082, 0.075), "Support\npredictions", COLORS["psc"], fs=6.7)
    box(ax, (0.575, 0.515), (0.082, 0.075), "Support\nresiduals", COLORS["psc"], fs=6.7)

    methods = [
        ("Residual\nPSC", 0.690, 0.645),
        ("Affine\nPSC", 0.690, 0.525),
        ("Shrinkage\nPSC", 0.690, 0.405),
    ]
    for label, x, y in methods:
        box(ax, (x, y), (0.080, 0.072), label, COLORS["white"], ec=COLORS["accent"], fs=6.8)

    box(ax, (0.575, 0.300), (0.195, 0.065), "No full model retraining", COLORS["psc"], fs=7.0, weight="bold")

    arrow(ax, (0.498, 0.388), (0.575, 0.670), lw=1.25, rad=0.14)
    arrow(ax, (0.616, 0.635), (0.616, 0.590), lw=1.10)
    arrow(ax, (0.657, 0.552), (0.690, 0.680), COLORS["accent"], 1.05, rad=0.18)
    arrow(ax, (0.657, 0.552), (0.690, 0.562), COLORS["accent"], 1.05)
    arrow(ax, (0.657, 0.552), (0.690, 0.440), COLORS["accent"], 1.05, rad=-0.18)


def draw_output(ax):
    region(ax, 0.830, 0.18, 0.135, 0.68, COLORS["output"])
    header(ax, 0.845, 0.815, "Output", COLORS["output"])

    box(ax, (0.846, 0.575), (0.100, 0.090), "Calibrated\nPPGR", COLORS["output"], fs=7.0, weight="bold")
    box(ax, (0.846, 0.455), (0.100, 0.065), "iAUC-2h", COLORS["white"], ec=COLORS["border"], fs=6.9)
    box(ax, (0.846, 0.350), (0.100, 0.065), "Bias\ncorrection", COLORS["white"], ec=COLORS["border"], fs=6.7)
    box(ax, (0.846, 0.245), (0.100, 0.065), "Shift-robust\nprediction", COLORS["white"], ec=COLORS["border"], fs=6.5)

    arrow(ax, (0.770, 0.562), (0.846, 0.620), lw=1.30)
    arrow(ax, (0.896, 0.575), (0.896, 0.520), COLORS["accent"], 1.05)
    arrow(ax, (0.896, 0.455), (0.896, 0.415), COLORS["accent"], 1.05)
    arrow(ax, (0.896, 0.350), (0.896, 0.310), COLORS["accent"], 1.05)


def main():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.8, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.12, 0.90)
    ax.axis("off")

    draw_input(ax)
    draw_model(ax)
    draw_psc(ax)
    draw_output(ax)

    arrow(ax, (0.250, 0.560), (0.280, 0.560), lw=1.35)
    arrow(ax, (0.525, 0.560), (0.555, 0.560), lw=1.35)
    arrow(ax, (0.800, 0.560), (0.830, 0.560), lw=1.35)

    ax.text(
        0.500,
        0.150,
        "Model-agnostic test-time PSC",
        ha="center",
        va="center",
        fontsize=7.2,
        color=COLORS["muted"],
    )

    for ext in ("png", "pdf", "svg"):
        out = OUT_BASE.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.04)
        else:
            fig.savefig(out, bbox_inches="tight", pad_inches=0.04)

    plt.close(fig)
    print(f"Saved: {OUT_BASE.with_suffix('.png')}")
    print(f"Saved: {OUT_BASE.with_suffix('.pdf')}")
    print(f"Saved: {OUT_BASE.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
