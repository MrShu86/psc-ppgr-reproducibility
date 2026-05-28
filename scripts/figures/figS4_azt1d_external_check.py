from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "final"
DATA_PATH = ROOT / "outputs_azt1d" / "robust_psc_check" / "azt1d_robust_psc_best_calibrated_vs_no_update.csv"
SUMMARY_PATH = ROOT / "outputs_azt1d" / "robust_psc_check" / "azt1d_robust_psc_best_by_shot.csv"

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
}

MODEL_ORDER = ["Ridge", "HistGradientBoosting", "RandomForest", "XGBoost"]
MODEL_COLORS = {
    "Ridge": COLORS["output"],
    "HistGradientBoosting": COLORS["blue"],
    "RandomForest": COLORS["event"],
    "XGBoost": COLORS["psc"],
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.9,
            "axes.edgecolor": COLORS["border"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def build_real_data():
    df = pd.read_csv(DATA_PATH)
    df["shot"] = pd.to_numeric(df["shot"], errors="coerce").astype(int)
    required = {"model", "shot", "no_update_RMSE", "best_calibrated_RMSE", "relative_RMSE_reduction_%"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")
    if SUMMARY_PATH.exists():
        summary = pd.read_csv(SUMMARY_PATH)
    else:
        summary = pd.DataFrame()
    return df, summary, False


def build_template_data():
    rows = []
    rng = np.random.default_rng(17)
    for model, base in zip(MODEL_ORDER, [3150, 2900, 3000, 2850]):
        for shot in [1, 3, 5, 10, 20, 30]:
            after = base - 60 * np.log1p(shot) + rng.normal(0, 25)
            rows.append(
                {
                    "model": model,
                    "shot": shot,
                    "no_update_RMSE": base + rng.normal(0, 20),
                    "best_calibrated_RMSE": after,
                    "relative_RMSE_reduction_%": (base - after) / base * 100,
                    "best_calibration": "mean_shrink_tau10",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(), True


def load_data():
    if DATA_PATH.exists():
        return build_real_data()
    # Required real columns: model, shot, no_update_RMSE, best_calibrated_RMSE, relative_RMSE_reduction_%.
    return build_template_data()


def draw():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, summary, template = load_data()
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    shots = sorted(df["shot"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.25), constrained_layout=True)

    ax = axes[0]
    selected_shot = 10 if 10 in shots else shots[min(len(shots) - 1, 2)]
    sub10 = df[df["shot"].eq(selected_shot)].copy()
    x = np.arange(len(MODEL_ORDER))
    w = 0.34
    before = [float(sub10[sub10["model"].eq(m)]["no_update_RMSE"].iloc[0]) if len(sub10[sub10["model"].eq(m)]) else np.nan for m in MODEL_ORDER]
    after = [float(sub10[sub10["model"].eq(m)]["best_calibrated_RMSE"].iloc[0]) if len(sub10[sub10["model"].eq(m)]) else np.nan for m in MODEL_ORDER]
    ax.bar(x - w / 2, before, width=w, color=COLORS["output"], edgecolor=COLORS["border"], linewidth=0.8, label="No update")
    ax.bar(x + w / 2, after, width=w, color=COLORS["blue"], edgecolor=COLORS["border"], linewidth=0.8, label="Best PSC")
    ax.set_title(f"A AZT1D RMSE at K = {selected_shot}", loc="left", fontweight="bold")
    ax.set_ylabel("RMSE of iAUC2h prediction (mg/dL min)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Ridge", "HGB", "RF", "XGB"])
    ax.grid(axis="y", color=COLORS["output"], lw=0.7, alpha=0.8)
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    for model in MODEL_ORDER:
        sub = df[df["model"].eq(model)].sort_values("shot")
        if sub.empty:
            continue
        ax.plot(
            sub["shot"],
            sub["relative_RMSE_reduction_%"],
            marker="o",
            lw=1.9,
            ms=4.5,
            label=model.replace("HistGradientBoosting", "HGB").replace("RandomForest", "RF").replace("XGBoost", "XGB"),
            color=MODEL_COLORS[model],
            markeredgecolor=COLORS["border"],
            markeredgewidth=0.5,
        )
    ax.axhline(0, color=COLORS["muted"], lw=0.9, ls="--")
    ax.set_title("B Reduction across support sizes", loc="left", fontweight="bold")
    ax.set_xlabel("Support meals K")
    ax.set_ylabel("RMSE reduction vs no update (%)")
    ax.set_xticks(shots)
    ax.grid(axis="y", color=COLORS["output"], lw=0.7, alpha=0.8)
    ax.legend(frameon=False, loc="best")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if not summary.empty:
        n_subjects = int(summary["n_subjects"].dropna().iloc[0]) if "n_subjects" in summary.columns and summary["n_subjects"].notna().any() else None
        if n_subjects:
            axes[1].text(
                0.98,
                0.92,
                f"AZT1D external check: {n_subjects} subjects",
                transform=axes[1].transAxes,
                ha="right",
                va="top",
                color=COLORS["muted"],
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
            )
    if template:
        fig.text(0.5, 0.02, "Template data - replace with real AZT1D robust PSC results", ha="center", color=COLORS["muted"])

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"FigS4_azt1d_external_check.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    df.to_csv(OUT_DIR / "FigS4_azt1d_external_check_source.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    draw()
