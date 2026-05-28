import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


LABEL_MAP = {
    "random_meal_split.csv": "Random",
    "cross_subject_split.csv": "Cross-subject",
    "cross_device_dexcom_to_libre.csv": "Dexcom→Libre",
    "cross_device_libre_to_dexcom.csv": "Libre→Dexcom",
    "cross_setting_mealtype_holdout_breakfast.csv": "Breakfast holdout",
}

MAIN_SPLITS = [
    "random_meal_split.csv",
    "cross_device_dexcom_to_libre.csv",
    "cross_device_libre_to_dexcom.csv",
    "cross_subject_split.csv",
    "cross_setting_mealtype_holdout_breakfast.csv",
]

ROBUSTNESS_SPLITS = [
    "cross_device_dexcom_to_libre.csv",
    "cross_device_libre_to_dexcom.csv",
    "cross_subject_split.csv",
    "cross_setting_mealtype_holdout_breakfast.csv",
]


def infer_method_and_split(filename: str):
    name = Path(filename).name.replace("_predictions.csv", "")

    # tabular_baseline_XGBoost_cross_subject_split_iauc_2h
    if name.startswith("tabular_baseline_"):
        rest = name.replace("tabular_baseline_", "")
        rest = rest.replace("_iauc_2h", "")
        split_candidates = [
            "random_meal_split",
            "cross_subject_split",
            "cross_device_dexcom_to_libre",
            "cross_device_libre_to_dexcom",
            "cross_setting_mealtype_holdout_breakfast",
        ]
        for split in split_candidates:
            suffix = "_" + split
            if rest.endswith(suffix):
                model = rest[:-len(suffix)]
                return f"Tabular-{model}", "tabular", split + ".csv"

    if name.startswith("meter_v1_"):
        rest = name.replace("meter_v1_", "").replace("_iauc_2h", "")
        parts = rest.split("_")
        encoder = parts[0]
        split_file = "_".join(parts[1:]) + ".csv"
        return "METER-v1", encoder, split_file

    if name.startswith("meter_v2_"):
        rest = name.replace("meter_v2_", "").replace("_iauc_2h", "")
        parts = rest.split("_")
        encoder = parts[0]
        split_file = "_".join(parts[1:]) + ".csv"
        return "METER-v2", encoder, split_file

    if name.startswith("sequence_baseline_"):
        rest = name.replace("sequence_baseline_", "").replace("_iauc_2h", "")
        parts = rest.split("_")
        encoder = parts[0]
        split_file = "_".join(parts[1:]) + ".csv"
        return "Sequence", encoder, split_file

    return "Unknown", "unknown", name + ".csv"


def normalize_model_label(method, encoder):
    if method.startswith("Tabular-"):
        return method.replace("Tabular-", "")
    if method == "Sequence":
        return encoder.upper()
    if method.startswith("METER"):
        return f"{method}-{encoder.upper()}"
    return f"{method}-{encoder}"


def pearson(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(y_true[ok], y_pred[ok])[0, 1])


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) == 0:
        return {
            "n": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Pearson": np.nan,
            "bias": np.nan,
            "abs_bias": np.nan,
            "median_abs_error": np.nan,
            "p90_abs_error": np.nan,
        }
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "Pearson": pearson(y_true, y_pred),
        "bias": float(np.mean(err)),
        "abs_bias": float(abs(np.mean(err))),
        "median_abs_error": float(np.median(np.abs(err))),
        "p90_abs_error": float(np.quantile(np.abs(err), 0.90)),
        "error_std": float(np.std(err, ddof=1)) if len(err) > 1 else np.nan,
    }


def residual_calibrate(pred_support, y_support, pred_query):
    if len(y_support) == 0:
        return pred_query, 0.0
    delta = float(np.nanmean(np.asarray(y_support) - np.asarray(pred_support)))
    return np.asarray(pred_query) + delta, delta


def affine_calibrate(pred_support, y_support, pred_query):
    pred_support = np.asarray(pred_support, dtype=float)
    y_support = np.asarray(y_support, dtype=float)
    pred_query = np.asarray(pred_query, dtype=float)
    ok = np.isfinite(pred_support) & np.isfinite(y_support)
    if ok.sum() < 3 or np.nanstd(pred_support[ok]) < 1e-8:
        y_cal, delta = residual_calibrate(pred_support, y_support, pred_query)
        return y_cal, np.nan, delta
    a, b = np.polyfit(pred_support[ok], y_support[ok], deg=1)
    return a * pred_query + b, float(a), float(b)


def merge_metadata(pred_df, metadata_path):
    if metadata_path is None:
        return pred_df
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return pred_df

    meta = pd.read_csv(metadata_path)
    keep_cols = [
        "event_id", "baseline_glucose", "setting_baseline_bin",
        "setting_activity_bin", "Carbs", "Calories", "Fat", "Protein", "Fiber"
    ]
    keep_cols = [c for c in keep_cols if c in meta.columns]
    if "event_id" not in keep_cols:
        return pred_df

    meta_small = meta[keep_cols].drop_duplicates("event_id")
    out = pred_df.merge(meta_small, on="event_id", how="left", suffixes=("", "_meta"))
    return out


def build_calibrated_query_predictions(pred_df, method, encoder, split_file, shots=(5,), calibration_modes=("support_residual_calibration",)):
    df = pred_df.copy()
    required = {"subject_id", "paired_event_id", "meal_timestamp", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction dataframe missing required columns: {missing}")

    df["meal_timestamp"] = pd.to_datetime(df["meal_timestamp"], errors="coerce")
    rows = []

    for subject_id, sub_all in df.groupby("subject_id"):
        pair_order = (
            sub_all.groupby("paired_event_id")["meal_timestamp"]
            .min()
            .sort_values()
            .index
            .tolist()
        )
        if len(pair_order) < 3:
            continue

        for k in shots:
            if len(pair_order) <= k + 1:
                continue

            support_pairs = set(pair_order[:k])
            query_pairs = set(pair_order[k:])

            support = sub_all[sub_all["paired_event_id"].isin(support_pairs)].copy()
            query = sub_all[sub_all["paired_event_id"].isin(query_pairs)].copy()

            if len(query) < 3:
                continue

            ps = support["y_pred"].to_numpy(dtype=float)
            ys = support["y_true"].to_numpy(dtype=float)
            pq = query["y_pred"].to_numpy(dtype=float)

            # Always include no-update query rows.
            q0 = query.copy()
            q0["method"] = method
            q0["encoder"] = encoder
            q0["model_label"] = normalize_model_label(method, encoder)
            q0["split_file"] = split_file
            q0["split_label"] = LABEL_MAP.get(split_file, split_file)
            q0["shot"] = k
            q0["calibration"] = "global_no_update" if k > 0 else "global_0shot"
            q0["y_pred_calibrated"] = q0["y_pred"]
            q0["support_delta"] = np.nan
            q0["affine_a"] = np.nan
            q0["affine_b"] = np.nan
            q0["is_query"] = True
            q0["n_support"] = len(support)
            q0["n_query_subject"] = len(query)
            rows.append(q0)

            for mode in calibration_modes:
                if k == 0:
                    continue
                if mode == "support_residual_calibration":
                    pred_cal, delta = residual_calibrate(ps, ys, pq)
                    a, b = np.nan, delta
                elif mode == "support_affine_calibration":
                    pred_cal, a, b = affine_calibrate(ps, ys, pq)
                    delta = np.nan
                else:
                    continue

                qc = query.copy()
                qc["method"] = method
                qc["encoder"] = encoder
                qc["model_label"] = normalize_model_label(method, encoder)
                qc["split_file"] = split_file
                qc["split_label"] = LABEL_MAP.get(split_file, split_file)
                qc["shot"] = k
                qc["calibration"] = mode
                qc["y_pred_calibrated"] = pred_cal
                qc["support_delta"] = delta
                qc["affine_a"] = a
                qc["affine_b"] = b
                qc["is_query"] = True
                qc["n_support"] = len(support)
                qc["n_query_subject"] = len(query)
                rows.append(qc)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["error_before"] = out["y_pred"] - out["y_true"]
    out["error_after"] = out["y_pred_calibrated"] - out["y_true"]
    out["abs_error_before"] = np.abs(out["error_before"])
    out["abs_error_after"] = np.abs(out["error_after"])
    return out


def summarize_query_rows(calibrated_rows):
    group_cols = ["method", "encoder", "model_label", "split_file", "split_label", "shot", "calibration"]
    rows = []
    for keys, sub in calibrated_rows.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        m = metrics(sub["y_true"], sub["y_pred_calibrated"])
        row.update(m)
        row["mean_y_true"] = float(sub["y_true"].mean())
        row["mean_y_pred_raw"] = float(sub["y_pred"].mean())
        row["mean_y_pred_calibrated"] = float(sub["y_pred_calibrated"].mean())
        row["raw_bias"] = float((sub["y_pred"] - sub["y_true"]).mean())
        row["calibrated_bias"] = float((sub["y_pred_calibrated"] - sub["y_true"]).mean())
        row["bias_reduction_abs"] = abs(row["raw_bias"]) - abs(row["calibrated_bias"])
        row["bias_reduction_%"] = (
            (abs(row["raw_bias"]) - abs(row["calibrated_bias"])) / abs(row["raw_bias"]) * 100
            if abs(row["raw_bias"]) > 1e-12 else np.nan
        )
        row["MAE_reduction"] = float(sub["abs_error_before"].mean() - sub["abs_error_after"].mean())
        row["MAE_reduction_%"] = (
            (sub["abs_error_before"].mean() - sub["abs_error_after"].mean()) /
            sub["abs_error_before"].mean() * 100
            if sub["abs_error_before"].mean() > 1e-12 else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_dimension(calibrated_rows, dimensions):
    rows = []
    for dim in dimensions:
        if dim not in calibrated_rows.columns:
            continue
        for keys, sub in calibrated_rows.groupby(["method", "encoder", "split_file", "split_label", "shot", "calibration", dim], dropna=False):
            method, encoder, split_file, split_label, shot, calibration, dim_value = keys
            if len(sub) < 3:
                continue
            row = {
                "dimension": dim,
                "dimension_value": dim_value,
                "method": method,
                "encoder": encoder,
                "model_label": normalize_model_label(method, encoder),
                "split_file": split_file,
                "split_label": split_label,
                "shot": shot,
                "calibration": calibration,
            }
            row.update(metrics(sub["y_true"], sub["y_pred_calibrated"]))
            row["raw_bias"] = float((sub["y_pred"] - sub["y_true"]).mean())
            row["calibrated_bias"] = float((sub["y_pred_calibrated"] - sub["y_true"]).mean())
            row["bias_reduction_abs"] = abs(row["raw_bias"]) - abs(row["calibrated_bias"])
            row["MAE_reduction"] = float(sub["abs_error_before"].mean() - sub["abs_error_after"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def select_prediction_files(results_dir, models):
    files = sorted(Path(results_dir).glob("*_predictions.csv"))
    selected = []

    for p in files:
        method, encoder, split_file = infer_method_and_split(p.name)
        model_label = normalize_model_label(method, encoder)

        if split_file not in MAIN_SPLITS:
            continue

        if models == ["all"]:
            selected.append((p, method, encoder, split_file))
            continue

        for m in models:
            if m.lower() in model_label.lower() or m.lower() in method.lower():
                selected.append((p, method, encoder, split_file))
                break

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/results")
    parser.add_argument("--output_dir", default="outputs/supplement_exp4")
    parser.add_argument("--metadata", default="outputs/events_metadata.csv")
    parser.add_argument("--shots", default="5,10")
    parser.add_argument("--calibrations", default="support_residual_calibration,support_affine_calibration")
    parser.add_argument(
        "--models",
        default="HistGradientBoosting,XGBoost,METER-v1",
        help="Comma-separated filters, or 'all'. Examples: HistGradientBoosting,XGBoost,METER-v1"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shots = tuple(int(x.strip()) for x in args.shots.split(",") if x.strip())
    calibrations = tuple(x.strip() for x in args.calibrations.split(",") if x.strip())
    models = [x.strip() for x in args.models.split(",") if x.strip()]

    selected = select_prediction_files(args.results_dir, models)
    if not selected:
        raise RuntimeError("No prediction files matched. Check --results_dir and --models.")

    all_rows = []
    skipped = []

    for p, method, encoder, split_file in selected:
        try:
            pred = pd.read_csv(p)
            pred = merge_metadata(pred, args.metadata)
            q = build_calibrated_query_predictions(
                pred,
                method=method,
                encoder=encoder,
                split_file=split_file,
                shots=shots,
                calibration_modes=calibrations,
            )
            if q.empty:
                skipped.append((p.name, "empty calibrated query rows"))
                continue
            q["source_file"] = p.name
            all_rows.append(q)
            print(f"[OK] {p.name}: {len(q)} calibrated query rows")
        except Exception as e:
            skipped.append((p.name, str(e)))

    if not all_rows:
        raise RuntimeError("No calibrated query rows generated.")

    allq = pd.concat(all_rows, ignore_index=True)

    # Save detailed calibrated predictions.
    detail_path = output_dir / "supplement_exp4_calibrated_query_predictions.csv"
    allq.to_csv(detail_path, index=False)

    summary = summarize_query_rows(allq)
    summary_path = output_dir / "supplement_exp4_bias_summary_by_method_split.csv"
    summary.to_csv(summary_path, index=False)

    # Dimension analysis.
    dims = ["device", "meal_type", "setting_baseline_bin", "setting_activity_bin"]
    dim_summary = summarize_by_dimension(allq, dims)
    dim_path = output_dir / "supplement_exp4_bias_summary_by_dimension.csv"
    dim_summary.to_csv(dim_path, index=False)

    # Subject-level bias before/after.
    subj_rows = []
    for keys, sub in allq.groupby(["method", "encoder", "model_label", "split_file", "split_label", "shot", "calibration", "subject_id"], dropna=False):
        method, encoder, model_label, split_file, split_label, shot, calibration, subject_id = keys
        if len(sub) < 2:
            continue
        row = {
            "method": method,
            "encoder": encoder,
            "model_label": model_label,
            "split_file": split_file,
            "split_label": split_label,
            "shot": shot,
            "calibration": calibration,
            "subject_id": subject_id,
            "n_query": len(sub),
            "raw_bias": float((sub["y_pred"] - sub["y_true"]).mean()),
            "calibrated_bias": float((sub["y_pred_calibrated"] - sub["y_true"]).mean()),
            "raw_RMSE": float(np.sqrt(np.mean((sub["y_pred"] - sub["y_true"]) ** 2))),
            "calibrated_RMSE": float(np.sqrt(np.mean((sub["y_pred_calibrated"] - sub["y_true"]) ** 2))),
            "raw_MAE": float(np.mean(np.abs(sub["y_pred"] - sub["y_true"]))),
            "calibrated_MAE": float(np.mean(np.abs(sub["y_pred_calibrated"] - sub["y_true"]))),
        }
        row["bias_abs_reduction"] = abs(row["raw_bias"]) - abs(row["calibrated_bias"])
        row["RMSE_reduction"] = row["raw_RMSE"] - row["calibrated_RMSE"]
        row["MAE_reduction"] = row["raw_MAE"] - row["calibrated_MAE"]
        subj_rows.append(row)

    subj = pd.DataFrame(subj_rows)
    subj_path = output_dir / "supplement_exp4_subject_level_bias_before_after.csv"
    subj.to_csv(subj_path, index=False)

    # Best bias-reduction view for each split and model.
    if not summary.empty:
        best_bias = summary.copy()
        best_bias = best_bias[best_bias["calibration"].ne("global_no_update")]
        if not best_bias.empty:
            best_bias = best_bias.loc[best_bias.groupby(["model_label", "split_file"])["bias_reduction_abs"].idxmax()]
            best_bias.to_csv(output_dir / "supplement_exp4_best_bias_reduction_by_model_split.csv", index=False)

    if skipped:
        pd.DataFrame(skipped, columns=["file", "reason"]).to_csv(output_dir / "supplement_exp4_skipped_files.csv", index=False)

    print(f"[OK] Saved calibrated predictions: {detail_path}")
    print(f"[OK] Saved bias summary: {summary_path}")
    print(f"[OK] Saved dimension summary: {dim_path}")
    print(f"[OK] Saved subject-level summary: {subj_path}")

    preview = summary[
        summary["split_file"].isin(ROBUSTNESS_SPLITS) &
        summary["calibration"].isin(["support_residual_calibration", "support_affine_calibration"])
    ].sort_values(["split_file", "model_label", "shot", "calibration"])
    cols = ["split_label", "model_label", "shot", "calibration", "RMSE", "MAE", "raw_bias", "calibrated_bias", "bias_reduction_abs", "MAE_reduction_%"]
    print("\n[BIAS SUMMARY PREVIEW]")
    print(preview[cols].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
