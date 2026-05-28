from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "final"
DATA_PATH = ROOT / "outputs" / "supplement_exp2" / "supplement_exp2_support_all_summary.csv"

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

PROTOCOL_ORDER = [
    "cross_device_dexcom_to_libre.csv",
    "cross_device_libre_to_dexcom.csv",
    "cross_subject_split.csv",
    "cross_setting_mealtype_holdout_breakfast.csv",
]
PROTOCOL_LABELS = {
    "cross_device_dexcom_to_libre.csv": "Dexcom -> Libre",
    "cross_device_libre_to_dexcom.csv": "Libre -> Dexcom",
    "cross_subject_split.csv": "Cross-subject",
    "cross_setting_mealtype_holdout_breakfast.csv": "Breakfast holdout",
}
SHOTS = [1, 3, 5, 10]


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
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def family_label(method):
    if str(method).startswith("Tabular-") and any(x in str(method) for x in ["XGBoost", "HistGradientBoosting"]):
        return "Best Tree-PSC"
    if str(method).startswith("METER"):
        return "Best METER-PSC"
    return None


def build_real_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["split_file"].isin(PROTOCOL_ORDER)].copy()
    df["shot"] = pd.to_numeric(df["shot"], errors="coerce").astype("Int64")
    df["family"] = df["method"].map(family_label)
    df = df[df["family"].notna()].copy()

    psc = df[
        df["personalization"].isin(["support_residual_calibration", "support_affine_calibration"])
        & df["shot"].isin(SHOTS)
    ].copy()
    psc = psc.sort_values("RMSE_mean").groupby(["family", "split_file", "shot"], as_index=False).first()

    base = df[df["personalization"].eq("global_no_update")].copy()
    base = base.sort_values("RMSE_mean").groupby(["family", "split_file", "shot"], as_index=False).first()
    merged = psc.merge(
        base[["family", "split_file", "shot", "RMSE_mean"]],
        on=["family", "split_file", "shot"],
        how="left",
        suffixes=("", "_base"),
    )
    merged["reduction"] = (merged["RMSE_mean_base"] - merged["RMSE_mean"]) / merged["RMSE_mean_base"] * 100.0
    return merged, False


def build_template_data():
    rows = []
    rng = np.random.default_rng(7)
    for family, base in [("Best Tree-PSC", 2550), ("Best METER-PSC", 2750)]:
        for split in PROTOCOL_ORDER:
            shift = PROTOCOL_ORDER.index(split) * 120
            for shot in SHOTS:
                rmse = base + shift - 150 * np.log1p(shot) + rng.normal(0, 40)
                rows.append(
                    {
                        "family": family,
                        "split_file": split,
                        "shot": shot,
                        "RMSE_mean": rmse,
                        "RMSE_mean_base": base + shift + 80,
                        "reduction": (base + shift + 80 - rmse) / (base + shift + 80) * 100,
                    }
                )
    return pd.DataFrame(rows), True


def load_data():
    if DATA_PATH.exists():
        return build_real_data()
    # Required real columns: method, split_file, personalization, shot, RMSE_mean.
    return build_template_data()


def line_summary(df, value_col):
    return (
        df.groupby(["family", "shot"])[value_col]
        .agg(mean="mean", std="std")
        .reset_index()
        .sort_values(["family", "shot"])
    )


def draw():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, template = load_data()
    rmse = line_summary(df, "RMSE_mean")
    red = line_summary(df, "reduction")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    specs = [
        (axes[0], rmse, "RMSE of iAUC2h prediction (mg/dL min)", "A Mean RMSE across shifted protocols"),
        (axes[1], red, "RMSE reduction vs no update (%)", "B Mean PSC gain across shifted protocols"),
    ]
    style = {
        "Best Tree-PSC": dict(color=COLORS["blue"], marker="o"),
        "Best METER-PSC": dict(color=COLORS["border"], marker="s"),
    }

    for ax, sub, ylabel, title in specs:
        for family, g in sub.groupby("family"):
            ax.errorbar(
                g["shot"],
                g["mean"],
                yerr=g["std"],
                lw=2.0,
                ms=5,
                capsize=3,
                label=family,
                **style[family],
            )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Support meals K")
        ax.set_ylabel(ylabel)
        ax.set_xticks(SHOTS)
        ax.grid(axis="y", color=COLORS["output"], lw=0.7, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].axhline(0, color=COLORS["muted"], lw=0.9, ls="--")
    axes[1].legend(frameon=False, loc="best")
    if template:
        fig.text(0.5, 0.02, "Template data - replace with real support calibration summary", ha="center", color=COLORS["muted"])

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"FigS1_k_shot_sensitivity.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    df.to_csv(OUT_DIR / "FigS1_k_shot_sensitivity_source.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    draw()
