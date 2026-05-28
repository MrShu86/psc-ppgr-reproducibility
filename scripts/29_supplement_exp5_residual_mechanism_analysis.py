import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


ROBUSTNESS_SPLITS = [
    "cross_device_dexcom_to_libre.csv",
    "cross_device_libre_to_dexcom.csv",
    "cross_subject_split.csv",
    "cross_setting_mealtype_holdout_breakfast.csv",
]


def safe_corr(x, y, method="spearman"):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 5:
        return np.nan, np.nan, int(ok.sum())
    if method == "spearman":
        r, p = stats.spearmanr(x[ok], y[ok])
    else:
        r, p = stats.pearsonr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def make_quantile_bins(s, q=3):
    try:
        return pd.qcut(pd.to_numeric(s, errors="coerce"), q=q, labels=["low", "mid", "high"], duplicates="drop")
    except Exception:
        return pd.Series([np.nan] * len(s), index=s.index)


def summarize_by_group(df, group_cols):
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = len(sub)
        row["raw_bias"] = float((sub["y_pred"] - sub["y_true"]).mean())
        row["calibrated_bias"] = float((sub["y_pred_calibrated"] - sub["y_true"]).mean())
        row["raw_MAE"] = float(np.abs(sub["y_pred"] - sub["y_true"]).mean())
        row["calibrated_MAE"] = float(np.abs(sub["y_pred_calibrated"] - sub["y_true"]).mean())
        row["raw_RMSE"] = float(np.sqrt(np.mean((sub["y_pred"] - sub["y_true"]) ** 2)))
        row["calibrated_RMSE"] = float(np.sqrt(np.mean((sub["y_pred_calibrated"] - sub["y_true"]) ** 2)))
        row["bias_abs_reduction"] = abs(row["raw_bias"]) - abs(row["calibrated_bias"])
        row["MAE_reduction"] = row["raw_MAE"] - row["calibrated_MAE"]
        row["RMSE_reduction"] = row["raw_RMSE"] - row["calibrated_RMSE"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrated_predictions", default="outputs/supplement_exp4/supplement_exp4_calibrated_query_predictions.csv")
    parser.add_argument("--output_dir", default="outputs/supplement_exp5")
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--calibration", default="support_residual_calibration")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.calibrated_predictions)
    df = df[
        df["shot"].eq(args.shot) &
        df["calibration"].eq(args.calibration) &
        df["split_file"].isin(ROBUSTNESS_SPLITS)
    ].copy()

    if df.empty:
        raise RuntimeError("No rows after filtering. Check shot/calibration or calibrated prediction file.")

    # Add residual quantities.
    df["raw_error"] = df["y_pred"] - df["y_true"]
    df["calibrated_error"] = df["y_pred_calibrated"] - df["y_true"]
    df["raw_abs_error"] = np.abs(df["raw_error"])
    df["calibrated_abs_error"] = np.abs(df["calibrated_error"])
    df["abs_error_reduction"] = df["raw_abs_error"] - df["calibrated_abs_error"]

    # Numeric residual correlations.
    candidate_numeric = [
        "baseline_glucose", "Carbs", "Calories", "Fat", "Protein", "Fiber",
        "support_delta", "n_support", "n_query_subject"
    ]
    corr_rows = []
    for col in candidate_numeric:
        if col not in df.columns:
            continue
        for (split_label, model_label), sub in df.groupby(["split_label", "model_label"]):
            for target_col in ["raw_error", "calibrated_error", "raw_abs_error", "calibrated_abs_error", "abs_error_reduction"]:
                spearman_r, spearman_p, n = safe_corr(sub[col], sub[target_col], "spearman")
                pearson_r, pearson_p, _ = safe_corr(sub[col], sub[target_col], "pearson")
                corr_rows.append({
                    "split_label": split_label,
                    "model_label": model_label,
                    "feature": col,
                    "residual_metric": target_col,
                    "n": n,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                })

    corr = pd.DataFrame(corr_rows)
    corr.to_csv(output_dir / "supplement_exp5_residual_feature_correlations.csv", index=False)

    # Create bins for numeric variables and summarize.
    for col in ["baseline_glucose", "Carbs", "Calories", "Fat", "Protein"]:
        if col in df.columns:
            df[f"{col}_bin"] = make_quantile_bins(df[col], q=3)

    group_cols_list = []
    for dim in ["device", "meal_type", "setting_baseline_bin", "setting_activity_bin",
                "baseline_glucose_bin", "Carbs_bin", "Calories_bin", "Fat_bin", "Protein_bin"]:
        if dim in df.columns:
            group_cols_list.append(["split_label", "model_label", dim])

    group_summaries = []
    for group_cols in group_cols_list:
        tmp = summarize_by_group(df, group_cols)
        tmp["dimension"] = group_cols[-1]
        tmp = tmp.rename(columns={group_cols[-1]: "dimension_value"})
        group_summaries.append(tmp)

    if group_summaries:
        dim_summary = pd.concat(group_summaries, ignore_index=True)
        dim_summary.to_csv(output_dir / "supplement_exp5_residual_group_summary.csv", index=False)

    # Top high-error examples before and after calibration.
    keep_cols = [
        "event_id", "paired_event_id", "subject_id", "device", "meal_type", "meal_timestamp",
        "split_label", "model_label", "y_true", "y_pred", "y_pred_calibrated",
        "raw_error", "calibrated_error", "raw_abs_error", "calibrated_abs_error",
        "baseline_glucose", "Carbs", "Calories", "Fat", "Protein", "Fiber",
        "setting_baseline_bin", "setting_activity_bin", "support_delta"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    top_raw = df.sort_values("raw_abs_error", ascending=False).head(200)[keep_cols]
    top_cal = df.sort_values("calibrated_abs_error", ascending=False).head(200)[keep_cols]
    top_improved = df.sort_values("abs_error_reduction", ascending=False).head(200)[keep_cols]
    top_worse = df.sort_values("abs_error_reduction", ascending=True).head(200)[keep_cols]

    top_raw.to_csv(output_dir / "supplement_exp5_top_raw_error_cases.csv", index=False)
    top_cal.to_csv(output_dir / "supplement_exp5_top_calibrated_error_cases.csv", index=False)
    top_improved.to_csv(output_dir / "supplement_exp5_top_improved_cases.csv", index=False)
    top_worse.to_csv(output_dir / "supplement_exp5_top_worsened_cases.csv", index=False)

    # Overall model/split summary for residual mechanism.
    overall = summarize_by_group(df, ["split_label", "model_label"])
    overall.to_csv(output_dir / "supplement_exp5_residual_mechanism_overall.csv", index=False)

    print(f"[OK] Saved residual mechanism analysis to {output_dir}")
    print("\n[Overall preview]")
    print(overall.sort_values(["split_label", "calibrated_RMSE"]).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
