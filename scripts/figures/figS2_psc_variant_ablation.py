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
    "cross_device_dexcom_to_libre.csv": "Dexcom\n-> Libre",
    "cross_device_libre_to_dexcom.csv": "Libre\n-> Dexcom",
    "cross_subject_split.csv": "Cross-\nsubject",
    "cross_setting_mealtype_holdout_breakfast.csv": "Breakfast\nholdout",
}
VARIANT_LABELS = {
    "global_no_update": "No update",
    "support_residual_calibration": "Residual PSC",
    "support_affine_calibration": "Affine PSC",
}
VARIANT_ORDER = ["global_no_update", "support_residual_calibration", "support_affine_calibration"]


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


def family_label(method):
    method = str(method)
    if method.startswith("Tabular-") and ("XGBoost" in method or "HistGradientBoosting" in method):
        return "Best Tree-PSC"
    if method.startswith("METER"):
        return "Best METER-PSC"
    return None


def build_real_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["split_file"].isin(PROTOCOL_ORDER)].copy()
    df["shot"] = pd.to_numeric(df["shot"], errors="coerce").astype("Int64")
    df = df[df["shot"].eq(5)].copy()
    df["family"] = df["method"].map(family_label)
    df = df[df["family"].notna() & df["personalization"].isin(VARIANT_ORDER)].copy()
    best = df.sort_values("RMSE_mean").groupby(["family", "split_file", "personalization"], as_index=False).first()
    return best, False


def build_template_data():
    rows = []
    rng = np.random.default_rng(11)
    for family, base in [("Best Tree-PSC", 2500), ("Best METER-PSC", 2700)]:
        for split in PROTOCOL_ORDER:
            shift = PROTOCOL_ORDER.index(split) * 180
            for variant, gain in zip(VARIANT_ORDER, [0, 350, 260]):
                rows.append(
                    {
                        "family": family,
                        "split_file": split,
                        "personalization": variant,
                        "RMSE_mean": base + shift - gain + rng.normal(0, 50),
                    }
                )
    return pd.DataFrame(rows), True


def load_data():
    if DATA_PATH.exists():
        return build_real_data()
    # Required real columns: method, split_file, personalization, shot, RMSE_mean.
    return build_template_data()


def draw_family(ax, df, family, title):
    sub = df[df["family"].eq(family)].copy()
    x = np.arange(len(PROTOCOL_ORDER))
    width = 0.24
    colors = {
        "global_no_update": COLORS["output"],
        "support_residual_calibration": COLORS["blue"],
        "support_affine_calibration": COLORS["psc"],
    }
    edge = COLORS["border"]
    for i, variant in enumerate(VARIANT_ORDER):
        vals = []
        for split in PROTOCOL_ORDER:
            m = sub[(sub["split_file"].eq(split)) & (sub["personalization"].eq(variant))]
            vals.append(float(m["RMSE_mean"].iloc[0]) if len(m) else np.nan)
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=VARIANT_LABELS[variant],
            color=colors[variant],
            edgecolor=edge,
            linewidth=0.8,
        )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([PROTOCOL_LABELS[s] for s in PROTOCOL_ORDER])
    ax.set_ylabel("RMSE of iAUC2h prediction (mg/dL min)")
    ax.grid(axis="y", color=COLORS["output"], lw=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, template = load_data()

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), sharey=True, constrained_layout=True)
    draw_family(axes[0], df, "Best Tree-PSC", "A Tree-based PSC variants (K = 5)")
    draw_family(axes[1], df, "Best METER-PSC", "B METER PSC variants (K = 5)")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    if template:
        fig.text(0.5, 0.02, "Template data - replace with real support calibration summary", ha="center", color=COLORS["muted"])

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"FigS2_psc_variant_ablation.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    df.to_csv(OUT_DIR / "FigS2_psc_variant_ablation_source.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    draw()
