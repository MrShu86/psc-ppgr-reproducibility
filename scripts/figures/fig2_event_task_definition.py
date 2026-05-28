from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
META_CSV = ROOT / "outputs" / "events_metadata.csv"
SEQ_NPZ = ROOT / "outputs" / "events_sequences.npz"
OUT_DIR = ROOT / "figures" / "final"
OUT_BASE = OUT_DIR / "Fig2_event_task_definition"

RNG = np.random.default_rng(42)


COLORS = {
    "data": "#D9EAF7",
    "event": "#DFF1E4",
    "shift": "#FCE5CD",
    "model": "#E8E1F5",
    "psc": "#D8F0F2",
    "output": "#ECEFF3",
    "blue": "#2C7FB8",
    "text": "#1F2D3D",
    "muted": "#566573",
    "border": "#34495E",
    "grid": "#DDE4EA",
    "white": "#FFFFFF",
    "green": "#2A9D8F",
    "orange": "#E99A3A",
    "purple": "#7A6BB7",
    "red": "#C75D5D",
    "beige": "#F8E8C8",
}

MEAL_COLORS = {
    "Breakfast": "#3B73B9",
    "Lunch": "#2A9D8F",
    "Dinner": "#E99A3A",
    "Snack": "#7A6BB7",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
        }
    )


def clean_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.8, length=3)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.55, alpha=0.75)


def panel_title(ax, letter, title):
    ax.text(
        -0.075,
        1.045,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        0.00,
        1.050,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["text"],
    )


def box(ax, xy, wh, text, fc, ec=None, fs=7.8, weight="normal", lw=1.05):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.014,rounding_size=0.030",
        facecolor=fc,
        edgecolor=ec or COLORS["border"],
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=COLORS["text"],
        fontweight=weight,
        linespacing=1.12,
    )
    return patch


def arrow(ax, start, end, color=None, lw=1.15, rad=0.0):
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


def load_real_data():
    """Real version expects:
    - outputs/events_metadata.csv with paired_event_id, device, meal_type, iauc_2h.
    - outputs/events_sequences.npz with arrays sequences and rel_grid.
    If either file is missing, only the representative curve/histogram are templated.
    """
    if not (META_CSV.exists() and SEQ_NPZ.exists()):
        return None, None, None
    meta = pd.read_csv(META_CSV)
    npz = np.load(SEQ_NPZ)
    return meta, npz["sequences"], npz["rel_grid"]


def smooth(y, window=3):
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    return np.convolve(np.pad(y, (pad, pad), mode="edge"), kernel, mode="valid")


def pick_representative_pair(meta, seq, rel_grid):
    work = meta.reset_index(drop=True).copy()
    work["_idx"] = np.arange(len(work))
    best = None
    best_score = np.inf

    for _, group in work.groupby("paired_event_id"):
        devices = group["device"].astype(str).str.lower()
        if not {"dexcom", "libre"}.issubset(set(devices)):
            continue
        dex = group[devices.eq("dexcom")].iloc[0]
        lib = group[devices.eq("libre")].iloc[0]
        idx_d = int(dex["_idx"])
        idx_l = int(lib["_idx"])
        mean_curve = np.nanmean(np.vstack([seq[idx_d], seq[idx_l]]), axis=0)
        pre = (rel_grid >= -30) & (rel_grid <= 0)
        post = (rel_grid >= 0) & (rel_grid <= 120)
        baseline = float(np.nanmean(mean_curve[pre]))
        delta = mean_curve[post] - baseline
        peak_rise = float(np.nanmax(delta))
        iauc = float(np.trapz(np.maximum(delta, 0), rel_grid[post]))
        shape_gap = float(np.nanmean(np.abs(seq[idx_d] - seq[idx_l])))
        if not (100 <= baseline <= 145 and 25 <= peak_rise <= 70 and 1200 <= iauc <= 5500):
            continue
        score = abs(peak_rise - 45) + abs(iauc - 3000) / 120 + shape_gap / 3
        if np.isfinite(score) and score < best_score:
            best = idx_d, idx_l
            best_score = score

    if best is not None:
        return best

    # Fallback still uses real events if a paired meal exists.
    for _, group in work.groupby("paired_event_id"):
        devices = group["device"].astype(str).str.lower()
        if {"dexcom", "libre"}.issubset(set(devices)):
            return int(group[devices.eq("dexcom")].iloc[0]["_idx"]), int(group[devices.eq("libre")].iloc[0]["_idx"])
    return None


def example_curve():
    """Template-only representative curve.
    Replace this with a real paired event from events_sequences.npz for final analysis.
    """
    t = np.arange(-30, 181, 5)
    baseline = 122 + 2 * np.sin((t + 20) / 20)
    response = 45 * np.exp(-0.5 * ((t - 55) / 26) ** 2)
    tail = 16 * np.exp(-0.5 * ((t - 125) / 42) ** 2)
    mean = baseline + response + tail
    dexcom = mean - 4 + 2 * np.sin(t / 17)
    libre = mean + 4 + 2 * np.cos(t / 21)
    return t, dexcom, libre


def draw_panel_a(ax):
    panel_title(ax, "A", "Event construction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.015, 0.70), (0.265, 0.16), "CGM streams\nLibre + Dexcom", COLORS["data"], ec=COLORS["blue"], fs=7.2)
    box(ax, (0.015, 0.49), (0.265, 0.16), "Meal records\nnutrition + type", COLORS["event"], ec=COLORS["green"], fs=7.2)
    box(ax, (0.015, 0.28), (0.265, 0.16), "Subject context\nclinical metadata", COLORS["model"], ec=COLORS["purple"], fs=6.9)

    box(ax, (0.375, 0.515), (0.255, 0.195), "Meal-centered\nwindowing\n-30 to +180 min", COLORS["shift"], ec=COLORS["orange"], fs=6.9, weight="bold")
    box(ax, (0.730, 0.535), (0.255, 0.185), "3370 events\n1700 paired meals\n45 subjects", COLORS["output"], fs=6.35, weight="bold")
    box(ax, (0.750, 0.290), (0.225, 0.135), "Libre: 1700\nDexcom: 1670", COLORS["output"], fs=7.0)

    arrow(ax, (0.280, 0.780), (0.375, 0.630))
    arrow(ax, (0.280, 0.570), (0.375, 0.610))
    arrow(ax, (0.280, 0.360), (0.375, 0.590))
    arrow(ax, (0.630, 0.612), (0.730, 0.628))
    arrow(ax, (0.858, 0.535), (0.858, 0.425))

    ax.text(0.395, 0.455, "5-min grid", fontsize=7.0, color=COLORS["muted"], ha="left")
    ax.text(0.395, 0.400, "quality filtering", fontsize=7.0, color=COLORS["muted"], ha="left")


def draw_panel_b(ax, meta, seq, rel_grid):
    panel_title(ax, "B", "PPGR target definition")

    if meta is not None and seq is not None and rel_grid is not None:
        pair = pick_representative_pair(meta, seq, rel_grid)
    else:
        pair = None

    if pair is not None:
        t = rel_grid
        dexcom = smooth(seq[pair[0]], 3)
        libre = smooth(seq[pair[1]], 3)
    else:
        t, dexcom, libre = example_curve()
        ax.text(0.03, 0.04, "TEMPLATE CURVE", transform=ax.transAxes, fontsize=7, color=COLORS["muted"])

    mean_curve = np.nanmean(np.vstack([dexcom, libre]), axis=0)
    pre = (t >= -30) & (t <= 0)
    post = (t >= 0) & (t <= 120)
    baseline = float(np.nanmean(mean_curve[pre]))
    delta = mean_curve - baseline
    peak_idx = int(np.nanargmax(np.where(post, delta, np.nan)))
    peak_t = float(t[peak_idx])
    peak_y = float(mean_curve[peak_idx])

    ax.axvspan(-30, 0, color=COLORS["beige"], alpha=0.42, linewidth=0)
    ax.axvspan(0, 120, color=COLORS["event"], alpha=0.35, linewidth=0)
    ax.axvline(0, color=COLORS["border"], lw=0.95, linestyle=(0, (2, 2)))
    ax.axhline(baseline, color=COLORS["muted"], lw=0.95, linestyle=(0, (4, 3)))

    ax.plot(t, dexcom, color="#3B73B9", lw=1.8, label="Dexcom")
    ax.plot(t, libre, color=COLORS["green"], lw=1.8, label="Libre")
    ax.plot(t, mean_curve, color=COLORS["border"], lw=1.8, label="Mean")
    dense_t = np.linspace(0, 120, 400)
    dense_mean = np.interp(dense_t, t, mean_curve)
    ax.fill_between(
        dense_t,
        baseline,
        dense_mean,
        where=dense_mean >= baseline,
        interpolate=True,
        color=COLORS["orange"],
        alpha=0.22,
        linewidth=0,
    )
    ax.scatter([peak_t], [peak_y], s=32, color=COLORS["red"], zorder=4)
    ax.annotate(
        "Peak rise",
        xy=(peak_t, peak_y),
        xytext=(peak_t + 24, peak_y + 8),
        fontsize=8,
        color=COLORS["red"],
        arrowprops=dict(arrowstyle="-|>", color=COLORS["red"], lw=1.0, shrinkA=2, shrinkB=3),
    )
    y_low, y_high = ax.get_ylim()
    label_box = dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.2)
    ax.text(-27, y_high - 8, "Pre-meal", color=COLORS["muted"], fontsize=8, bbox=label_box)
    ax.text(18, y_low + 8, "Post-meal response", color=COLORS["green"], fontsize=8, bbox=label_box)
    ax.text(122, baseline + 2.2, "Baseline", color=COLORS["muted"], fontsize=7.6, bbox=label_box)
    ax.text(42, baseline + 5, "iAUC2h", color=COLORS["orange"], fontsize=8)

    ax.set_xlim(-30, 180)
    y_min = min(95, np.nanmin(mean_curve) - 10)
    y_max = max(185, np.nanmax(mean_curve) + 15)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([-30, 0, 60, 120, 180])
    ax.set_xlabel("Time from meal onset (min)")
    ax.set_ylabel("Glucose (mg/dL)")
    clean_axis(ax, "y")
    ax.legend(frameon=False, loc="upper right", handlelength=2.5)


def draw_panel_c(fig, spec, meta):
    outer = fig.add_subplot(spec)
    outer.axis("off")
    panel_title(outer, "C", "Event and target distributions")

    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[1.0, 1.15], wspace=0.42)
    ax_bar = fig.add_subplot(inner[0, 0])
    ax_hist = fig.add_subplot(inner[0, 1])

    meals = ["Breakfast", "Lunch", "Dinner", "Snack"]
    counts = [832, 869, 983, 686]
    x = np.arange(len(meals))
    ax_bar.bar(x, counts, color=[MEAL_COLORS[m] for m in meals], width=0.62, edgecolor="white", linewidth=0.7)
    for xi, val in zip(x, counts):
        ax_bar.text(xi, val + 28, str(val), ha="center", va="bottom", fontsize=7.8, color=COLORS["text"])
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(meals, rotation=25, ha="right")
    ax_bar.set_ylabel("Events")
    ax_bar.set_ylim(0, 1120)
    clean_axis(ax_bar, "y")

    if meta is not None and "iauc_2h" in meta.columns:
        vals = pd.to_numeric(meta["iauc_2h"], errors="coerce").dropna().to_numpy()
    else:
        # Template only. Required real column: outputs/events_metadata.csv::iauc_2h.
        vals = RNG.gamma(shape=1.35, scale=2100, size=3370)
        ax_hist.text(0.03, 0.92, "TEMPLATE", transform=ax_hist.transAxes, fontsize=7, color=COLORS["muted"])
    vals = vals[np.isfinite(vals)]
    vals = vals[vals >= 0]
    ax_hist.hist(vals, bins=24, color="#F3B3B3", edgecolor=COLORS["red"], linewidth=0.55, alpha=0.42)
    ax_hist.set_xlabel("iAUC2h (mg/dL*min)")
    ax_hist.set_ylabel("Count", labelpad=7)
    clean_axis(ax_hist, "y")


def draw_panel_d(ax):
    panel_title(ax, "D", "Shift-aware mining task")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.015, 0.665), (0.295, 0.15), "Input event\nCGM + meal\n+ context", COLORS["data"], ec=COLORS["blue"], fs=6.8, weight="bold")
    box(ax, (0.365, 0.665), (0.275, 0.15), "Base predictor\n$f_\\theta(x)$", COLORS["model"], ec=COLORS["green"], fs=7.4, weight="bold")
    box(ax, (0.700, 0.665), (0.285, 0.15), "Output PPGR\niAUC2h", COLORS["output"], ec=COLORS["orange"], fs=7.4, weight="bold")
    arrow(ax, (0.310, 0.74), (0.365, 0.74))
    arrow(ax, (0.640, 0.74), (0.700, 0.74))

    shifts = [
        ("Device shift", "Dexcom <-> Libre", "#3B73B9"),
        ("Subject shift", "seen subjects -> held-out subject", "#5B8E3E"),
        ("Setting shift", "meal type / baseline / activity", "#7A6BB7"),
    ]
    y0 = 0.52
    for i, (name, desc, color) in enumerate(shifts):
        y = y0 - i * 0.105
        ax.add_patch(Rectangle((0.08, y - 0.022), 0.020, 0.044, facecolor=color, edgecolor=color, linewidth=0.8))
        ax.text(0.13, y, name, ha="left", va="center", fontsize=8.2, color=COLORS["text"], fontweight="bold")
        ax.text(0.47, y, desc, ha="left", va="center", fontsize=8.0, color=COLORS["muted"])

    ax.text(0.50, 0.205, "Robust inference under shifted domains", ha="center", fontsize=7.3, color=COLORS["red"])
    box(ax, (0.16, 0.07), (0.25, 0.075), "Source support", COLORS["white"], ec=COLORS["muted"], fs=8.0)
    box(ax, (0.58, 0.07), (0.25, 0.075), "Shifted query", COLORS["white"], ec=COLORS["muted"], fs=8.0)
    arrow(ax, (0.41, 0.108), (0.58, 0.108), color=COLORS["red"], lw=1.3)


def main():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta, seq, rel_grid = load_real_data()

    fig = plt.figure(figsize=(11.2, 7.1))
    gs = GridSpec(2, 2, figure=fig, hspace=0.43, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b, meta, seq, rel_grid)
    draw_panel_c(fig, gs[1, 0], meta)
    draw_panel_d(ax_d)

    for ext in ("png", "svg", "pdf"):
        out = OUT_BASE.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.05)
        else:
            fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"Saved: {OUT_BASE.with_suffix('.png')}")
    print(f"Saved: {OUT_BASE.with_suffix('.pdf')}")
    print(f"Saved: {OUT_BASE.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
