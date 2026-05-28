from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "final"
DATA_PATH = ROOT / "outputs" / "supplement_exp5" / "supplement_exp5_feature_importance_overall.csv"

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


def clean_feature(name):
    name = str(name)
    replacements = {
        "subject_": "",
        "Calories (Activity)_pre30_mean": "Activity calories",
        "Fasting GLU - PDL (Lab)": "Fasting glucose",
        "A1c PDL (Lab)": "HbA1c",
        "baseline_glucose": "Baseline glucose",
        "pre_slope": "Pre-meal slope",
        "meal_type": "Meal type",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name[:34]


def build_real_data():
    df = pd.read_csv(DATA_PATH)
    required = {"model", "feature", "mean_importance"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")
    if "std_importance" not in df.columns:
        df["std_importance"] = 0.0
    return df, False


def build_template_data():
    features = ["Carbs", "Baseline glucose", "Pre-meal slope", "Meal type", "HbA1c", "Protein", "Age", "Fiber"]
    rows = []
    rng = np.random.default_rng(13)
    for model in ["HistGradientBoosting", "XGBoost"]:
        for i, feature in enumerate(features):
            rows.append(
                {
                    "model": model,
                    "feature": feature,
                    "mean_importance": 420 / (i + 1) + rng.normal(0, 12),
                    "std_importance": 25 + 8 * rng.random(),
                    "n_splits": 4,
                }
            )
    return pd.DataFrame(rows), True


def load_data():
    if DATA_PATH.exists():
        return build_real_data()
    # Required real columns: model, feature, mean_importance, optional std_importance.
    return build_template_data()


def draw_panel(ax, df, model, title, color):
    sub = df[df["model"].eq(model)].copy()
    if sub.empty:
        ax.text(0.5, 0.5, f"No {model} data", transform=ax.transAxes, ha="center", va="center", color=COLORS["muted"])
        ax.set_axis_off()
        return
    sub = sub.sort_values("mean_importance", ascending=False).head(10).iloc[::-1].copy()
    y = np.arange(len(sub))
    ax.barh(
        y,
        sub["mean_importance"],
        xerr=sub["std_importance"],
        color=color,
        edgecolor=COLORS["border"],
        linewidth=0.8,
        error_kw=dict(ecolor=COLORS["muted"], lw=0.8, capsize=2),
    )
    ax.set_yticks(y)
    ax.set_yticklabels([clean_feature(v) for v in sub["feature"]])
    ax.set_xlabel("Permutation importance (RMSE increase)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", color=COLORS["output"], lw=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, template = load_data()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), constrained_layout=True)
    draw_panel(axes[0], df, "HistGradientBoosting", "A HistGradientBoosting", COLORS["blue"])
    draw_panel(axes[1], df, "XGBoost", "B XGBoost", COLORS["psc"])
    if template:
        fig.text(0.5, 0.02, "Template data - replace with real feature importance outputs", ha="center", color=COLORS["muted"])

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"FigS3_feature_importance.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    df.to_csv(OUT_DIR / "FigS3_feature_importance_source.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    draw()
