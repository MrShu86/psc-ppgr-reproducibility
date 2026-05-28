from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "final"
OUT_BASE = OUT_DIR / "Fig5_main_performance"
STANDARD_RESULTS = ROOT / "results" / "model_results.csv"
RESULTS_DIR = ROOT / "outputs" / "results"
PSC_RESULTS = ROOT / "outputs" / "supplement_exp4" / "supplement_exp4_best_bias_reduction_by_model_split.csv"

RNG = np.random.default_rng(42)
BOOTSTRAP_N = 1000


# Palette copied from Fig. 2. Keep Fig. 5 within the established manuscript
# color set; do not introduce additional model colors.
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

MODEL_STYLE = {
    "Ridge": ("#566573", "o"),
    "HGB": ("#2A9D8F", "s"),
    "XGBoost": ("#E99A3A", "^"),
    "TCN": ("#7A6BB7", "D"),
    "METER-v1": ("#3B73B9", "P"),
    "HGB/XGBoost + PSC": ("#C75D5D", "*"),
    "METER-PSC": ("#2C7FB8", "X"),
}

PROTOCOL_ORDER = [
    "Random",
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


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
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


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.8, length=3, color=COLORS["border"])
    ax.spines["left"].set_color(COLORS["border"])
    ax.spines["bottom"].set_color(COLORS["border"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.75)


def panel_title(ax, letter, title):
    ax.text(
        -0.075,
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


def protocol_from_split(split):
    return SPLIT_TO_PROTOCOL.get(str(split), None)


def normalize_standard_results(path):
    df = pd.read_csv(path)
    model_col = first_existing_column(df, ["model", "model_label", "method"])
    split_col = first_existing_column(df, ["protocol", "split", "split_file"])
    rmse_col = first_existing_column(df, ["rmse", "RMSE"])
    mae_col = first_existing_column(df, ["mae", "MAE"])
    r2_col = first_existing_column(df, ["r2", "R2"])
    pearson_col = first_existing_column(df, ["pearson", "Pearson"])
    ci_low_col = first_existing_column(df, ["rmse_ci_low", "ci_low"])
    ci_high_col = first_existing_column(df, ["rmse_ci_high", "ci_high"])

    if model_col is None or split_col is None or rmse_col is None:
        raise ValueError("results/model_results.csv must include model, split/protocol, and rmse columns.")

    out = pd.DataFrame(
        {
            "model_raw": df[model_col].astype(str),
            "protocol_raw": df[split_col].astype(str),
            "RMSE": pd.to_numeric(df[rmse_col], errors="coerce"),
            "MAE": pd.to_numeric(df[mae_col], errors="coerce") if mae_col else np.nan,
            "R2": pd.to_numeric(df[r2_col], errors="coerce") if r2_col else np.nan,
            "Pearson": pd.to_numeric(df[pearson_col], errors="coerce") if pearson_col else np.nan,
            "rmse_ci_low": pd.to_numeric(df[ci_low_col], errors="coerce") if ci_low_col else np.nan,
            "rmse_ci_high": pd.to_numeric(df[ci_high_col], errors="coerce") if ci_high_col else np.nan,
        }
    )
    out["protocol"] = out["protocol_raw"].map(protocol_from_split).fillna(out["protocol_raw"])
    out["source"] = str(path)
    return out


def read_metric_file(path):
    df = pd.read_csv(path)
    if not {"split_file", "model", "RMSE"}.issubset(df.columns):
        return pd.DataFrame()
    out = df.copy()
    out["protocol"] = out["split_file"].map(protocol_from_split)
    out = out[out["protocol"].isin(PROTOCOL_ORDER)].copy()
    out["source"] = path.name
    return out


def collect_base_results():
    rows = []
    if (RESULTS_DIR / "strong_tabular_prediction_results_iauc_2h.csv").exists():
        tab = pd.read_csv(RESULTS_DIR / "strong_tabular_prediction_results_iauc_2h.csv")
        tab["protocol"] = tab["split_file"].map(protocol_from_split)
        tab = tab[tab["protocol"].isin(PROTOCOL_ORDER)].copy()
        tab["source"] = "strong_tabular_prediction_results_iauc_2h.csv"
        rows.append(tab)

    for pattern in [
        "sequence_baseline_tcn_*_iauc_2h.csv",
        "meter_v1_tcn_*_iauc_2h.csv",
        "meter_v2_tcn_*_iauc_2h.csv",
    ]:
        for path in RESULTS_DIR.glob(pattern):
            df = read_metric_file(path)
            if len(df):
                rows.append(df)

    if not rows:
        return pd.DataFrame()

    raw = pd.concat(rows, ignore_index=True)

    def display_model(name):
        name = str(name)
        if name == "HistGradientBoosting":
            return "HGB"
        if name == "tcn":
            return "TCN"
        if name.startswith("meter_v1"):
            return "METER-v1"
        if name.startswith("meter_v2"):
            return "METER-v2"
        if name in {"Ridge", "XGBoost"}:
            return name
        return None

    raw["display_model"] = raw["model"].map(display_model)
    return raw


def collect_psc_results():
    if not PSC_RESULTS.exists():
        return pd.DataFrame()
    df = pd.read_csv(PSC_RESULTS)
    df = df[df["split_file"].isin(SPLIT_TO_PROTOCOL)].copy()
    df = df[df["calibration"].astype(str).str.contains("support", case=False, na=False)].copy()
    if df.empty:
        return pd.DataFrame()
    df["protocol"] = df["split_file"].map(protocol_from_split)

    tree = df[df["method"].isin(["Tabular-HistGradientBoosting", "Tabular-XGBoost"])].copy()
    if len(tree):
        tree = tree.sort_values(["split_file", "RMSE"]).groupby("split_file", as_index=False).head(1).copy()
        tree["display_model"] = "HGB/XGBoost + PSC"
        tree["psc_family"] = "Best Tree-PSC"
        tree["model"] = tree["method"] + " " + tree["calibration"].astype(str) + " K=" + tree["shot"].astype(str)

    meter = df[df["method"].isin(["METER-v1", "METER-v2"])].copy()
    if len(meter):
        meter = meter.sort_values(["split_file", "RMSE"]).groupby("split_file", as_index=False).head(1).copy()
        meter["display_model"] = "METER-PSC"
        meter["psc_family"] = "Best METER-PSC"
        meter["model"] = meter["method"] + " " + meter["calibration"].astype(str) + " K=" + meter["shot"].astype(str)

    out = pd.concat([tree, meter], ignore_index=True, sort=False)
    out["source"] = PSC_RESULTS.name
    return out


def build_template_results():
    # Template only. Replace with results/model_results.csv containing:
    # model, split/protocol, rmse, mae, r2, pearson, and optional RMSE CIs.
    models = ["Ridge", "HGB", "XGBoost", "TCN", "METER-v1", "HGB/XGBoost + PSC", "METER-PSC"]
    protocol_base = {
        "Random": 1900,
        "Dexcom -> Libre": 2200,
        "Libre -> Dexcom": 2350,
        "Cross-subject": 2550,
        "Breakfast holdout": 3300,
    }
    model_effect = {
        "Ridge": 520,
        "HGB": 120,
        "XGBoost": 40,
        "TCN": 180,
        "METER-v1": 90,
        "HGB/XGBoost + PSC": -330,
        "METER-PSC": -260,
    }
    rows = []
    for protocol in PROTOCOL_ORDER:
        for model in models:
            rmse = protocol_base[protocol] + model_effect[model] + RNG.normal(0, 55)
            rows.append(
                {
                    "protocol": protocol,
                    "display_model": model,
                    "model": model,
                    "RMSE": rmse,
                    "MAE": np.nan,
                    "R2": np.nan,
                    "Pearson": np.nan,
                    "rmse_ci_low": rmse - 120,
                    "rmse_ci_high": rmse + 120,
                    "source": "template",
                }
            )
    return pd.DataFrame(rows), True


def load_results():
    if STANDARD_RESULTS.exists():
        df = normalize_standard_results(STANDARD_RESULTS)
        df["display_model"] = df["model_raw"].replace({"HistGradientBoosting": "HGB", "TCN": "TCN"})
        return df, False, str(STANDARD_RESULTS)

    base = collect_base_results()
    psc = collect_psc_results()
    if base.empty and psc.empty:
        return (*build_template_results(), "template")

    full = pd.concat([base, psc], ignore_index=True, sort=False)
    full = full[full["protocol"].isin(PROTOCOL_ORDER)].copy()
    full = full.rename(columns={"display_model": "display_model"})
    full["RMSE"] = pd.to_numeric(full["RMSE"], errors="coerce")
    full["MAE"] = pd.to_numeric(full.get("MAE", np.nan), errors="coerce")
    full["R2"] = pd.to_numeric(full.get("R2", np.nan), errors="coerce")
    full["Pearson"] = pd.to_numeric(full.get("Pearson", np.nan), errors="coerce")
    full = full.dropna(subset=["RMSE", "display_model", "protocol"])
    return full, False, "outputs/results + outputs/supplement_exp4"


def prepare_plot_data(df):
    keep = list(MODEL_STYLE.keys())
    df = df[df["display_model"].isin(keep)].copy()
    df["model_order"] = df["display_model"].map({m: i for i, m in enumerate(keep)})
    df["protocol_order"] = df["protocol"].map({p: i for i, p in enumerate(PROTOCOL_ORDER)})
    df = df.sort_values(["protocol_order", "model_order", "RMSE"])
    # If duplicate rows remain for a model/protocol, keep the lowest RMSE real result.
    df = df.groupby(["protocol", "display_model"], as_index=False).first()
    return df


def bootstrap_rmse_ci(pred_df, pred_col="y_pred", n_boot=BOOTSTRAP_N):
    y_true = pd.to_numeric(pred_df["y_true"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(pred_df[pred_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    pred_df = pred_df.loc[valid].copy()
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) < 4:
        return np.nan, np.nan

    if "subject_id" in pred_df.columns and pred_df["subject_id"].nunique() >= 2:
        subjects = pred_df["subject_id"].astype(str).to_numpy()
        unique_subjects = np.unique(subjects)
        by_subject = {s: np.flatnonzero(subjects == s) for s in unique_subjects}
        boot = []
        for _ in range(n_boot):
            sampled_subjects = RNG.choice(unique_subjects, size=len(unique_subjects), replace=True)
            idx = np.concatenate([by_subject[s] for s in sampled_subjects])
            err = y_true[idx] - y_pred[idx]
            boot.append(np.sqrt(np.mean(err * err)))
    else:
        boot = []
        n = len(y_true)
        for _ in range(n_boot):
            idx = RNG.integers(0, n, size=n)
            err = y_true[idx] - y_pred[idx]
            boot.append(np.sqrt(np.mean(err * err)))
    return np.percentile(boot, [2.5, 97.5])


def prediction_path_for_row(row):
    split = str(row.get("split_file", ""))
    if split not in SPLIT_TO_PROTOCOL:
        return None
    stem = split.replace(".csv", "")
    model = str(row.get("display_model", ""))
    if model == "Ridge":
        return RESULTS_DIR / f"tabular_baseline_Ridge_{stem}_iauc_2h_predictions.csv"
    if model == "HGB":
        return RESULTS_DIR / f"tabular_baseline_HistGradientBoosting_{stem}_iauc_2h_predictions.csv"
    if model == "XGBoost":
        return RESULTS_DIR / f"tabular_baseline_XGBoost_{stem}_iauc_2h_predictions.csv"
    if model == "METER-v1":
        return RESULTS_DIR / f"meter_v1_tcn_{stem}_iauc_2h_predictions.csv"
    if model == "METER-v2":
        return RESULTS_DIR / f"meter_v2_tcn_{stem}_iauc_2h_predictions.csv"
    return None


def attach_rmse_ci(plot_df):
    plot_df = plot_df.copy()
    ci_low = []
    ci_high = []
    ci_source = []
    calibrated = None

    for _, row in plot_df.iterrows():
        model = str(row.get("display_model", ""))
        if model in {"HGB/XGBoost + PSC", "METER-PSC"}:
            if calibrated is None and PSC_RESULTS.exists():
                pred_path = PSC_RESULTS.parent / "supplement_exp4_calibrated_query_predictions.csv"
                calibrated = pd.read_csv(pred_path)
            if calibrated is None:
                ci_low.append(np.nan)
                ci_high.append(np.nan)
                ci_source.append("missing calibrated prediction table")
                continue
            shot_value = pd.to_numeric(pd.Series([row.get("shot", np.nan)]), errors="coerce").iloc[0]
            pred_shot = pd.to_numeric(calibrated["shot"], errors="coerce")
            pred = calibrated[
                calibrated["method"].astype(str).eq(str(row.get("method", "")))
                & calibrated["split_file"].astype(str).eq(str(row.get("split_file", "")))
                & pred_shot.eq(shot_value)
                & calibrated["calibration"].astype(str).eq(str(row.get("calibration", "")))
            ].copy()
            if pred.empty or "y_pred_calibrated" not in pred.columns:
                ci_low.append(np.nan)
                ci_high.append(np.nan)
                ci_source.append("missing PSC prediction rows")
                continue
            low, high = bootstrap_rmse_ci(pred, pred_col="y_pred_calibrated")
            ci_low.append(low)
            ci_high.append(high)
            ci_source.append("subject-cluster bootstrap from calibrated queries")
            continue

        pred_path = prediction_path_for_row(row)
        if pred_path is None or not pred_path.exists():
            ci_low.append(np.nan)
            ci_high.append(np.nan)
            ci_source.append("prediction table unavailable")
            continue
        pred = pd.read_csv(pred_path)
        if "y_true" not in pred.columns or "y_pred" not in pred.columns:
            ci_low.append(np.nan)
            ci_high.append(np.nan)
            ci_source.append("prediction columns unavailable")
            continue
        low, high = bootstrap_rmse_ci(pred, pred_col="y_pred")
        ci_low.append(low)
        ci_high.append(high)
        ci_source.append(f"subject-cluster bootstrap from {pred_path.name}")

    plot_df["rmse_ci_low"] = ci_low
    plot_df["rmse_ci_high"] = ci_high
    plot_df["rmse_ci_source"] = ci_source
    return plot_df


def comparator_for_psc(full_df, psc_row):
    protocol = str(psc_row.get("protocol", ""))
    method = str(psc_row.get("method", ""))
    if method == "Tabular-HistGradientBoosting":
        base = full_df[
            full_df["protocol"].eq(protocol)
            & full_df["display_model"].eq("HGB")
        ].copy()
    elif method == "Tabular-XGBoost":
        base = full_df[
            full_df["protocol"].eq(protocol)
            & full_df["display_model"].eq("XGBoost")
        ].copy()
    elif method == "METER-v2":
        base = full_df[
            full_df["protocol"].eq(protocol)
            & full_df["model"].astype(str).str.startswith("meter_v2")
        ].copy()
    else:
        base = full_df[
            full_df["protocol"].eq(protocol)
            & full_df["model"].astype(str).str.startswith("meter_v1")
        ].copy()

    if base.empty:
        return None
    return base.sort_values("RMSE").iloc[0]


def compute_psc_reduction(full_df, plot_df):
    rows = []
    for protocol in PROTOCOL_ORDER:
        for display_model, family in [
            ("HGB/XGBoost + PSC", "Best Tree-PSC"),
            ("METER-PSC", "Best METER-PSC"),
        ]:
            psc = plot_df[(plot_df["protocol"].eq(protocol)) & (plot_df["display_model"].eq(display_model))]
            if psc.empty:
                continue
            psc = psc.iloc[0]
            base = comparator_for_psc(full_df, psc)
            if base is None:
                continue

            base_rmse = float(base["RMSE"])
            psc_rmse = float(psc["RMSE"])
            reduction = 100.0 * (base_rmse - psc_rmse) / base_rmse
            rows.append(
                {
                    "protocol": protocol,
                    "psc_family": family,
                    "psc_model": str(psc.get("model", display_model)),
                    "baseline_model": str(base.get("model", base.get("display_model", ""))),
                    "baseline_type": "corresponding uncalibrated base predictor",
                    "baseline_RMSE": base_rmse,
                    "psc_RMSE": psc_rmse,
                    "RMSE_reduction_percent": reduction,
                }
            )
    return pd.DataFrame(rows)


def plot_performance(ax, df, template=False):
    panel_title(ax, "A", "Representative base models and PSC-enhanced models")
    x_base = np.arange(len(PROTOCOL_ORDER), dtype=float)
    models = list(MODEL_STYLE.keys())
    offsets = np.linspace(-0.25, 0.25, len(models))

    for idx, model in enumerate(models):
        sub = df[df["display_model"].eq(model)].copy()
        y = []
        xs = []
        yerr_lower = []
        yerr_upper = []
        for p_i, protocol in enumerate(PROTOCOL_ORDER):
            row = sub[sub["protocol"].eq(protocol)]
            if row.empty:
                y.append(np.nan)
                xs.append(x_base[p_i] + offsets[idx])
                yerr_lower.append(np.nan)
                yerr_upper.append(np.nan)
                continue
            r = row.iloc[0]
            y.append(float(r["RMSE"]))
            xs.append(x_base[p_i] + offsets[idx])
            low = r.get("rmse_ci_low", np.nan)
            high = r.get("rmse_ci_high", np.nan)
            if pd.notna(low) and pd.notna(high):
                yerr_lower.append(float(r["RMSE"] - low))
                yerr_upper.append(float(high - r["RMSE"]))
            else:
                yerr_lower.append(np.nan)
                yerr_upper.append(np.nan)

        color, marker = MODEL_STYLE[model]
        if np.isfinite(yerr_lower).any() and np.isfinite(yerr_upper).any():
            ax.errorbar(
                xs,
                y,
                yerr=[yerr_lower, yerr_upper],
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=0.85,
                capsize=2.0,
                markersize=5.1 if "PSC" not in model else 7.0,
                markeredgecolor=COLORS["white"],
                markeredgewidth=0.55,
                label=model,
                zorder=4,
            )
        else:
            ax.scatter(
                xs,
                y,
                s=30 if "PSC" not in model else 58,
                marker=marker,
                color=color,
                edgecolor=COLORS["white"],
                linewidth=0.55,
                label=model,
                zorder=4,
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels(PROTOCOL_ORDER)
    ax.set_ylabel(r"RMSE for iAUC2h (mg/dL$\cdot$min)")
    ax.set_xlabel("Evaluation protocol")
    ax.set_xlim(-0.55, len(PROTOCOL_ORDER) - 0.45)
    ymax = np.nanmax(df["RMSE"].to_numpy(dtype=float)) * 1.15
    ax.set_ylim(0, max(ymax, 1000))
    clean_axis(ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_STYLE[model][1],
            color="none",
            markerfacecolor=MODEL_STYLE[model][0],
            markeredgecolor=COLORS["white"],
            markeredgewidth=0.55,
            markersize=5.8 if "PSC" not in model else 7.2,
            label=model,
        )
        for model in MODEL_STYLE
    ]
    ax.legend(
        handles=legend_handles,
        ncol=7,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.22),
        columnspacing=0.95,
        handlelength=1.0,
        handletextpad=0.35,
    )
    if template:
        ax.text(0.01, 0.97, "TEMPLATE", transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=COLORS["muted"])
    ax.text(
        0.995,
        0.070,
        "PSC models: best support-calibrated tree variant and best support-calibrated METER variant",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=COLORS["muted"],
        bbox=dict(boxstyle="round,pad=0.20", fc=COLORS["white"], ec="none", alpha=0.78),
    )


def plot_reduction(ax, reduction_df, template=False):
    panel_title(ax, "B", "Relative RMSE reduction by PSC")
    x = np.arange(len(PROTOCOL_ORDER))
    groups = [
        ("Best Tree-PSC", -0.16, COLORS["psc"], COLORS["red"]),
        ("Best METER-PSC", 0.16, COLORS["shift"], COLORS["blue"]),
    ]
    ax.axhline(0, color=COLORS["border"], linewidth=0.9)
    all_values = []
    for family, offset_x, fill, edge in groups:
        values = []
        for protocol in PROTOCOL_ORDER:
            row = reduction_df[
                reduction_df["protocol"].eq(protocol)
                & reduction_df["psc_family"].eq(family)
            ]
            values.append(np.nan if row.empty else float(row.iloc[0]["RMSE_reduction_percent"]))
        all_values.extend([v for v in values if np.isfinite(v)])
        bars = ax.bar(
            x + offset_x,
            values,
            width=0.28,
            color=fill,
            edgecolor=edge,
            linewidth=1.0,
            label=family,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            va = "bottom" if value >= 0 else "top"
            offset_y = 1.0 if value >= 0 else -1.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset_y,
                f"{value:.1f}%",
                ha="center",
                va=va,
                fontsize=7.2,
                color=COLORS["text"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(PROTOCOL_ORDER)
    ax.set_ylabel("RMSE reduction by PSC (%)")
    ax.set_xlabel("Evaluation protocol")
    finite = np.asarray(all_values, dtype=float)
    if len(finite):
        y_min = min(0, np.nanmin(finite) - 6)
        y_max = max(10, np.nanmax(finite) + 8)
        ax.set_ylim(y_min, y_max)
    clean_axis(ax)
    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.995, 1.18),
        ncol=2,
        handlelength=1.2,
        columnspacing=1.0,
    )
    ax.text(
        0.01,
        0.94,
        "Comparator: corresponding uncalibrated base predictor",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["muted"],
    )
    if template:
        ax.text(0.01, 0.97, "TEMPLATE", transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=COLORS["muted"])


def main():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = load_results()
    if len(loaded) == 3:
        df, template, source = loaded
    else:
        df, template = loaded
        source = "template"
    plot_df = prepare_plot_data(df)
    plot_df = attach_rmse_ci(plot_df)
    reduction_df = compute_psc_reduction(df, plot_df)

    source_csv = OUT_DIR / "Fig5_main_performance_source.csv"
    reduction_csv = OUT_DIR / "Fig5_psc_reduction_source.csv"
    plot_df.to_csv(source_csv, index=False)
    reduction_df.to_csv(reduction_csv, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.2), gridspec_kw={"height_ratios": [1.45, 1.0]})
    plot_performance(axes[0], plot_df, template=template)
    plot_reduction(axes[1], reduction_df, template=template)
    fig.subplots_adjust(left=0.075, right=0.992, top=0.885, bottom=0.115, hspace=0.55)

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
    print(f"Reduction table: {reduction_csv}")
    print(f"Data source: {source}")


if __name__ == "__main__":
    main()
