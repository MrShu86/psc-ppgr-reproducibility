from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "final"
OUT_BASE = OUT_DIR / "Fig6_psc_robustness_bias"

STANDARD_RESULTS = ROOT / "results" / "support_calibration_results.csv"
BIAS_SUMMARY = ROOT / "outputs" / "supplement_exp4" / "supplement_exp4_bias_summary_by_method_split.csv"
BEST_BIAS = ROOT / "outputs" / "supplement_exp4" / "supplement_exp4_best_bias_reduction_by_model_split.csv"
METER_TESTS = ROOT / "outputs" / "supplement_exp3" / "supplement_exp3_support_vs_no_update_tests.csv"
TABULAR_TESTS = ROOT / "outputs" / "supplement_exp3" / "supplement_exp3_tabular_support_vs_no_update_tests.csv"

RNG = np.random.default_rng(42)


# Palette copied from Fig. 2 to keep the figure set visually consistent.
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
    "meal_blue": "#3B73B9",
}

PROTOCOL_ORDER = [
    "Dexcom -> Libre",
    "Libre -> Dexcom",
    "Cross-subject",
    "Breakfast holdout",
]

SPLIT_TO_PROTOCOL = {
    "random_meal_split.csv": "Random",
    "cross_device_dexcom_to_libre.csv": "Dexcom -> Libre",
    "cross_device_libre_to_dexcom.csv": "Libre -> Dexcom",
    "cross_subject_split.csv": "Cross-subject",
    "cross_setting_mealtype_holdout_breakfast.csv": "Breakfast holdout",
}

FAMILY_STYLE = {
    "Best Tree-PSC": {"color": COLORS["red"], "marker": "o", "fill": COLORS["psc"]},
    "Best METER-PSC": {"color": COLORS["blue"], "marker": "s", "fill": COLORS["shift"]},
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
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
    ax.tick_params(width=0.8, length=3, color=COLORS["border"])
    ax.spines["left"].set_color(COLORS["border"])
    ax.spines["bottom"].set_color(COLORS["border"])
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.55, alpha=0.75)


def panel_title(ax, letter, title):
    ax.text(
        -0.115,
        1.055,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        0.000,
        1.062,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.4,
        fontweight="bold",
        color=COLORS["text"],
    )


def first_existing_column(df, candidates):
    lower_to_original = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]
    return None


def significance_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def build_template_data():
    # Template only. Replace with results/support_calibration_results.csv containing
    # split/protocol, model, rmse_before, rmse_after, bias_before, and bias_after.
    rows = []
    for protocol in PROTOCOL_ORDER:
        for family in FAMILY_STYLE:
            before_rmse = RNG.normal(2700, 450)
            after_rmse = before_rmse * RNG.uniform(0.72, 0.92)
            before_bias = RNG.normal(0, 850)
            after_bias = before_bias * RNG.uniform(0.05, 0.25)
            rows.append(
                {
                    "protocol": protocol,
                    "psc_family": family,
                    "method": family,
                    "calibration": "support_calibration",
                    "shot": 5,
                    "rmse_before": before_rmse,
                    "rmse_after": after_rmse,
                    "bias_before": before_bias,
                    "bias_after": after_bias,
                    "p_value": np.nan,
                    "source": "template",
                }
            )
    return pd.DataFrame(rows), True, "template"


def load_standard_results(path):
    df = pd.read_csv(path)
    split_col = first_existing_column(df, ["protocol", "split", "split_file"])
    model_col = first_existing_column(df, ["model", "method", "model_label"])
    before_rmse = first_existing_column(df, ["rmse_before", "no_psc_rmse", "rmse_no_psc", "no_psc"])
    after_rmse = first_existing_column(df, ["rmse_after", "psc_rmse", "rmse_psc", "psc"])
    before_bias = first_existing_column(df, ["bias_before", "raw_bias", "bias_no_psc"])
    after_bias = first_existing_column(df, ["bias_after", "calibrated_bias", "bias_psc"])
    p_col = first_existing_column(df, ["p_value", "wilcoxon_p", "paired_t_p"])

    required = [split_col, model_col, before_rmse, after_rmse, before_bias, after_bias]
    if any(col is None for col in required):
        return build_template_data()

    out = pd.DataFrame(
        {
            "protocol": df[split_col].astype(str).map(SPLIT_TO_PROTOCOL).fillna(df[split_col].astype(str)),
            "psc_family": df[model_col].astype(str),
            "method": df[model_col].astype(str),
            "calibration": "support_calibration",
            "shot": np.nan,
            "rmse_before": pd.to_numeric(df[before_rmse], errors="coerce"),
            "rmse_after": pd.to_numeric(df[after_rmse], errors="coerce"),
            "bias_before": pd.to_numeric(df[before_bias], errors="coerce"),
            "bias_after": pd.to_numeric(df[after_bias], errors="coerce"),
            "p_value": pd.to_numeric(df[p_col], errors="coerce") if p_col else np.nan,
            "source": str(path),
        }
    )
    out = out[out["protocol"].isin(PROTOCOL_ORDER)].dropna(subset=["rmse_before", "rmse_after", "bias_before", "bias_after"])
    return out, False, str(path)


def collect_selected_psc_rows():
    best = pd.read_csv(BEST_BIAS)
    best = best[best["split_file"].isin(SPLIT_TO_PROTOCOL)].copy()
    best = best[best["calibration"].astype(str).str.contains("support", case=False, na=False)].copy()

    tree = best[best["method"].isin(["Tabular-HistGradientBoosting", "Tabular-XGBoost"])].copy()
    tree = tree.sort_values(["split_file", "RMSE"]).groupby("split_file", as_index=False).head(1)
    tree["psc_family"] = "Best Tree-PSC"

    meter = best[best["method"].isin(["METER-v1", "METER-v2"])].copy()
    meter = meter.sort_values(["split_file", "RMSE"]).groupby("split_file", as_index=False).head(1)
    meter["psc_family"] = "Best METER-PSC"

    selected = pd.concat([tree, meter], ignore_index=True, sort=False)
    selected["protocol"] = selected["split_file"].map(SPLIT_TO_PROTOCOL)
    return selected


def load_test_tables():
    frames = []
    if METER_TESTS.exists():
        frames.append(pd.read_csv(METER_TESTS))
    if TABULAR_TESTS.exists():
        frames.append(pd.read_csv(TABULAR_TESTS))
    if not frames:
        return pd.DataFrame()
    tests = pd.concat(frames, ignore_index=True, sort=False)
    return tests


def load_real_results():
    if not (BEST_BIAS.exists() and BIAS_SUMMARY.exists()):
        return build_template_data()

    selected = collect_selected_psc_rows()
    summary = pd.read_csv(BIAS_SUMMARY)
    tests = load_test_tables()
    rows = []

    for _, row in selected.iterrows():
        shot_value = pd.to_numeric(pd.Series([row["shot"]]), errors="coerce").iloc[0]
        baseline = summary[
            summary["method"].astype(str).eq(str(row["method"]))
            & summary["split_file"].astype(str).eq(str(row["split_file"]))
            & pd.to_numeric(summary["shot"], errors="coerce").eq(shot_value)
            & summary["calibration"].astype(str).eq("global_no_update")
        ].copy()
        if baseline.empty:
            continue
        baseline = baseline.iloc[0]

        p_value = np.nan
        if len(tests):
            match = tests[
                tests["method"].astype(str).eq(str(row["method"]))
                & tests["split_file"].astype(str).eq(str(row["split_file"]))
                & pd.to_numeric(tests["shot"], errors="coerce").eq(shot_value)
                & tests["tested"].astype(str).eq(str(row["calibration"]))
            ].copy()
            if len(match) and "wilcoxon_p" in match.columns:
                p_value = pd.to_numeric(match.iloc[0]["wilcoxon_p"], errors="coerce")

        rows.append(
            {
                "protocol": row["protocol"],
                "psc_family": row["psc_family"],
                "method": row["method"],
                "model_label": row.get("model_label", ""),
                "calibration": row["calibration"],
                "shot": shot_value,
                "rmse_before": float(baseline["RMSE"]),
                "rmse_after": float(row["RMSE"]),
                "bias_before": float(row["raw_bias"]),
                "bias_after": float(row["calibrated_bias"]),
                "p_value": p_value,
                "source": f"{BEST_BIAS.name} + {BIAS_SUMMARY.name}",
            }
        )

    if not rows:
        return build_template_data()
    out = pd.DataFrame(rows)
    out = out[out["protocol"].isin(PROTOCOL_ORDER)].copy()
    return out, False, "outputs/supplement_exp4 + outputs/supplement_exp3"


def load_results():
    if STANDARD_RESULTS.exists():
        return load_standard_results(STANDARD_RESULTS)
    return load_real_results()


def add_pair_lines(ax, df, y_before, y_after, ylabel, zero=False, annotate_sig=False):
    x0, x1 = 0.0, 1.0
    offsets = {"Best Tree-PSC": -0.035, "Best METER-PSC": 0.035}
    protocol_offsets = {
        protocol: delta
        for protocol, delta in zip(PROTOCOL_ORDER, np.linspace(-0.018, 0.018, len(PROTOCOL_ORDER)))
    }

    for _, row in df.iterrows():
        family = row["psc_family"]
        style = FAMILY_STYLE.get(family, FAMILY_STYLE["Best Tree-PSC"])
        dx = offsets.get(family, 0.0) + protocol_offsets.get(row["protocol"], 0.0)
        before = float(row[y_before])
        after = float(row[y_after])
        xs = [x0 + dx, x1 + dx]
        ys = [before, after]

        ax.plot(xs, ys, color=COLORS["grid"], linewidth=1.15, alpha=0.95, zorder=1)
        ax.scatter(
            xs[0],
            ys[0],
            s=24,
            marker=style["marker"],
            facecolor=COLORS["white"],
            edgecolor=COLORS["muted"],
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter(
            xs[1],
            ys[1],
            s=34,
            marker=style["marker"],
            facecolor=style["fill"],
            edgecolor=style["color"],
            linewidth=1.0,
            zorder=4,
        )
        if annotate_sig:
            stars = significance_stars(row.get("p_value", np.nan))
            if stars:
                ax.text(xs[1] + 0.035, ys[1], stars, ha="left", va="center", fontsize=7.2, color=style["color"])

    if zero:
        ax.axhline(0, color=COLORS["border"], linewidth=0.9, linestyle=(0, (4, 2)), zorder=0)
    ax.set_xlim(-0.22, 1.22)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before PSC", "After PSC"])
    ax.set_ylabel(ylabel)
    clean_axis(ax, "y")


def draw_panel_a(ax, df, template):
    panel_title(ax, "A", "Error reduction")
    add_pair_lines(
        ax,
        df,
        y_before="rmse_before",
        y_after="rmse_after",
        ylabel=r"RMSE for iAUC2h (mg/dL$\cdot$min)",
        zero=False,
        annotate_sig=True,
    )
    max_y = max(df["rmse_before"].max(), df["rmse_after"].max()) * 1.12
    ax.set_ylim(0, max_y)
    if template:
        ax.text(0.02, 0.96, "TEMPLATE", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=COLORS["muted"])


def draw_panel_b(ax, df, template):
    panel_title(ax, "B", "Bias correction")
    add_pair_lines(
        ax,
        df,
        y_before="bias_before",
        y_after="bias_after",
        ylabel=r"Prediction bias (mg/dL$\cdot$min)",
        zero=True,
        annotate_sig=False,
    )
    bound = np.nanmax(np.abs(df[["bias_before", "bias_after"]].to_numpy(dtype=float))) * 1.18
    ax.set_ylim(-bound, bound)
    if template:
        ax.text(0.02, 0.96, "TEMPLATE", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=COLORS["muted"])


def main():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, template, source = load_results()
    df["protocol_order"] = df["protocol"].map({p: i for i, p in enumerate(PROTOCOL_ORDER)})
    df["family_order"] = df["psc_family"].map({"Best Tree-PSC": 0, "Best METER-PSC": 1})
    df = df.sort_values(["protocol_order", "family_order"]).reset_index(drop=True)

    source_csv = OUT_DIR / "Fig6_psc_robustness_bias_source.csv"
    df.to_csv(source_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    draw_panel_a(axes[0], df, template)
    draw_panel_b(axes[1], df, template)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            color=COLORS["grid"],
            markerfacecolor=style["fill"],
            markeredgecolor=style["color"],
            markeredgewidth=1.0,
            linewidth=1.1,
            markersize=5.7,
            label=family,
        )
        for family, style in FAMILY_STYLE.items()
    ]
    fig.legend(
        handles=legend_handles,
        ncol=2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.02),
        columnspacing=1.2,
        handlelength=1.5,
    )
    fig.text(
        0.985,
        0.045,
        "Each line represents one shift-specific PSC implementation",
        ha="right",
        va="bottom",
        fontsize=7.3,
        color=COLORS["muted"],
    )
    fig.text(
        0.080,
        0.045,
        "* p < 0.05; ** p < 0.01; *** p < 0.001",
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.080, right=0.990, top=0.840, bottom=0.200, wspace=0.34)

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
    print(f"Source table: {source_csv}")
    print(f"Data source: {source}")


if __name__ == "__main__":
    main()
