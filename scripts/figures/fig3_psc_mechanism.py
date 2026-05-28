from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "final"
OUT_BASE = OUT_DIR / "Fig3_psc_mechanism"

META_CSV = ROOT / "outputs" / "events_metadata.csv"
SEQ_NPZ = ROOT / "outputs" / "events_sequences.npz"
CAL_PRED_CSV = ROOT / "outputs" / "supplement_exp4" / "supplement_exp4_calibrated_query_predictions.csv"
RESULTS_DIR = ROOT / "outputs" / "results"


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
    "green": "#2A9D8F",
    "orange": "#E99A3A",
    "red": "#C75D5D",
    "purple": "#7A6BB7",
    "white": "#FFFFFF",
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


def clean_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.8, length=3)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.55, alpha=0.75)


def box(ax, xy, wh, label, fc, ec=None, fs=7.4, weight="normal", lw=1.05):
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
        label,
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


def smooth(y, window=3):
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    pad = window // 2
    return np.convolve(np.pad(y, (pad, pad), mode="edge"), kernel, mode="valid")


def load_data():
    meta = pd.read_csv(META_CSV)
    arrays = np.load(SEQ_NPZ)
    seq = arrays["sequences"].astype(float)
    rel_grid = arrays["rel_grid"].astype(float)
    return meta, seq, rel_grid


def select_real_case():
    pred = pd.read_csv(CAL_PRED_CSV)
    query = pred[
        pred["is_query"].astype(str).str.lower().eq("true")
        & pred["method"].eq("METER-v2")
        & pred["calibration"].eq("support_residual_calibration")
        & pred["shot"].eq(5)
    ].copy()
    query["improvement"] = query["abs_error_before"] - query["abs_error_after"]
    query = query[
        query["improvement"].between(1000, 3000)
        & query["y_true"].between(800, 5000)
        & query["y_pred"].between(0, 8000)
    ].copy()
    query["score"] = (query["improvement"] - 2000).abs() + 0.10 * query["abs_error_after"]
    chosen = query.sort_values("score").iloc[0]

    # Support residuals are real model residuals from the same prediction source file.
    source_path = RESULTS_DIR / str(chosen["source_file"])
    source = pd.read_csv(source_path)
    source["meal_timestamp"] = pd.to_datetime(source["meal_timestamp"], errors="coerce")
    source = source.sort_values("meal_timestamp")

    subject = str(chosen["subject_id"])
    sub = source[source["subject_id"].astype(str).eq(subject)].copy()
    pair_order = sub.groupby("paired_event_id")["meal_timestamp"].min().sort_values().index.tolist()
    support_pairs = set(pair_order[: int(chosen["shot"])])
    support = sub[sub["paired_event_id"].isin(support_pairs)].sort_values("meal_timestamp").head(5).copy()
    return chosen, support, source_path.name


def meta_index_lookup(meta):
    return {str(event_id): i for i, event_id in enumerate(meta["event_id"].astype(str))}


def draw_panel_a(ax):
    panel_title(ax, "A", "Base event-centered PPGR predictor")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.04, 0.68), (0.20, 0.13), "Pre-meal\nCGM", COLORS["data"], ec=COLORS["blue"], fs=7.4, weight="bold")
    box(ax, (0.04, 0.49), (0.20, 0.13), "Meal\nfeatures", COLORS["event"], ec=COLORS["green"], fs=7.4, weight="bold")
    box(ax, (0.04, 0.30), (0.20, 0.13), "Subject\ncontext", COLORS["model"], ec=COLORS["purple"], fs=7.2, weight="bold")

    box(ax, (0.36, 0.53), (0.20, 0.15), "Event encoder\n$z=h_{\\theta}(x)$", COLORS["output"], ec=COLORS["muted"], fs=7.2, weight="bold")
    box(ax, (0.72, 0.53), (0.22, 0.15), "Base predictor\n$\\hat{y}=f_{\\theta}(z)$", COLORS["shift"], ec=COLORS["orange"], fs=7.2, weight="bold")

    for y in (0.745, 0.555, 0.365):
        arrow(ax, (0.24, y), (0.36, 0.605))
    arrow(ax, (0.56, 0.605), (0.72, 0.605))

    box(
        ax,
        (0.31, 0.220),
        (0.65, 0.145),
        "Support-calibrated inference\nQuery labels are held out",
        COLORS["white"],
        ec=COLORS["grid"],
        fs=6.7,
    )
    arrow(ax, (0.82, 0.53), (0.63, 0.355), color=COLORS["orange"], lw=1.1)
    ax.text(0.50, 0.13, "Shared event model across devices, subjects, and settings", ha="center", fontsize=7.0, color=COLORS["muted"])


def plot_mini_curve(ax, rel_grid, curve):
    curve = smooth(curve, 3)
    ax.axvspan(-30, 0, color="#F8E8C8", alpha=0.38, linewidth=0)
    ax.axvspan(0, 120, color=COLORS["event"], alpha=0.28, linewidth=0)
    ax.axvline(0, color=COLORS["muted"], lw=0.85, ls=(0, (2, 2)))
    ax.plot(rel_grid, curve, color="#3B73B9", lw=1.8)
    ymin, ymax = np.nanmin(curve), np.nanmax(curve)
    span = max(ymax - ymin, 20)
    ax.set_xlim(-30, 180)
    ax.set_ylim(ymin - 0.18 * span, ymax + 0.18 * span)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def draw_panel_b(ax, meta, seq, rel_grid, support, query_row):
    panel_title(ax, "B", "Personalized support set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    lookup = meta_index_lookup(meta)
    positions = [
        (0.06, 0.58, 0.23, 0.20),
        (0.38, 0.58, 0.23, 0.20),
        (0.70, 0.58, 0.23, 0.20),
        (0.22, 0.30, 0.23, 0.20),
        (0.54, 0.30, 0.23, 0.20),
    ]
    # These five mini-curves are real CGMacros event sequences from
    # outputs/events_sequences.npz, matched by event_id through outputs/events_metadata.csv.
    for i, (_, row) in enumerate(support.iterrows()):
        event_id = str(row["event_id"])
        if event_id not in lookup or i >= len(positions):
            continue
        mini = ax.inset_axes(positions[i])
        plot_mini_curve(mini, rel_grid, seq[lookup[event_id]])
        meal = str(row.get("meal_type", "meal")).capitalize()
        mini.set_title(f"S{i + 1} {meal}", fontsize=6.1, pad=1)

    subject = str(query_row["subject_id"])
    ax.text(0.08, 0.18, f"Subject: {subject}", ha="left", va="center", fontsize=6.6, color=COLORS["muted"])
    box(
        ax,
        (0.04, 0.045),
        (0.76, 0.130),
        "Support: first K = 5 observed meals\nQuery: subsequent meals",
        COLORS["white"],
        ec=COLORS["grid"],
        fs=6.3,
    )
    arrow(ax, (0.80, 0.110), (0.88, 0.110), color=COLORS["muted"], lw=1.0)
    ax.text(0.89, 0.110, "residuals\nin C", ha="left", va="center", fontsize=6.1, color=COLORS["muted"])


def draw_panel_c(ax, support):
    panel_title(ax, "C", "Support-based calibration rule")
    support = support.copy()
    # Residual bars are real residuals, y_i - yhat_i, from the selected support meals.
    support["residual"] = pd.to_numeric(support["y_true"], errors="coerce") - pd.to_numeric(support["y_pred"], errors="coerce")
    residuals = support["residual"].astype(float).to_numpy()
    delta_s = float(np.nanmean(residuals))

    x = np.arange(len(residuals))
    bar_colors = [COLORS["green"] if value >= 0 else COLORS["red"] for value in residuals]
    ax.bar(x, residuals, color=bar_colors, width=0.60, edgecolor="white", linewidth=0.7, alpha=0.88)
    ax.axhline(0, color=COLORS["border"], lw=0.9)
    ax.axhline(delta_s, color=COLORS["orange"], lw=1.6, ls=(0, (4, 2)), label=r"$\Delta_s$")
    ax.text(len(residuals) - 0.15, delta_s + 260, rf"$\Delta_s$ = {delta_s:.0f}", ha="right", color=COLORS["orange"], fontsize=8.0, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i + 1}" for i in x])
    ax.set_ylabel(r"Residual $y-\hat{y}$")
    ax.set_ylim(min(-3500, np.nanmin(residuals) - 500), max(np.nanmax(residuals) + 900, delta_s + 1200))
    clean_axis(ax, "y")

    ax.legend(frameon=False, loc="upper left")


def draw_panel_d(ax, query_row):
    panel_title(ax, "D", "Query prediction after PSC")
    true = float(query_row["y_true"])
    raw = float(query_row["y_pred"])
    calibrated = float(query_row["y_pred_calibrated"])
    raw_err = abs(raw - true)
    cal_err = abs(calibrated - true)

    values = [true, raw, calibrated]
    labels = ["True", "Base\nprediction", "PSC\nprediction"]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    x = np.arange(3)
    bars = ax.bar(x, values, width=0.56, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("iAUC2h (mg/dL*min)")
    ax.set_ylim(0, max(values) * 1.20)
    clean_axis(ax, "y")

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.035, f"{value:.0f}", ha="center", va="bottom", fontsize=7.2)

    ax.text(
        0.54,
        0.82,
        f"Selected query case\n$|y-\\hat{{y}}|$: {raw_err:.0f} -> {cal_err:.0f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color=COLORS["muted"],
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLORS["grid"], alpha=0.92),
    )


def main():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta, seq, rel_grid = load_data()
    query_row, support, source_name = select_real_case()

    fig = plt.figure(figsize=(11.2, 7.1))
    gs = GridSpec(2, 2, figure=fig, hspace=0.43, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b, meta, seq, rel_grid, support, query_row)
    draw_panel_c(ax_c, support)
    draw_panel_d(ax_d, query_row)

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
    print(f"Case source: {source_name}")
    print(f"Query event: {query_row['event_id']}")


if __name__ == "__main__":
    main()
