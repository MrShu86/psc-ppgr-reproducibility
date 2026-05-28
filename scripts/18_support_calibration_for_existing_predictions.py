import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Pearson": np.nan}
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "Pearson": pearson(y_true, y_pred),
    }


def infer_method_and_split(filename: str):
    name = Path(filename).name.replace("_predictions.csv", "")

    if name.startswith("meter_v1_"):
        method = "METER-v1"
        rest = name.replace("meter_v1_", "")
    elif name.startswith("meter_v2_"):
        method = "METER-v2"
        rest = name.replace("meter_v2_", "")
    elif name.startswith("sequence_baseline_"):
        method = "Sequence"
        rest = name.replace("sequence_baseline_", "")
    else:
        method = "Unknown"
        rest = name

    rest = rest.replace("_iauc_2h", "")
    parts = rest.split("_")
    if parts and parts[0] in {"tcn", "gru", "mlp", "transformer"}:
        encoder = parts[0]
        split_stem = "_".join(parts[1:])
    else:
        encoder = "unknown"
        split_stem = rest

    split_file = split_stem + ".csv"
    return method, encoder, split_file


def residual_calibrate(pred_support, y_support, pred_query):
    if len(y_support) == 0:
        return pred_query
    return np.asarray(pred_query) + float(np.nanmean(np.asarray(y_support) - np.asarray(pred_support)))


def affine_calibrate(pred_support, y_support, pred_query):
    pred_support = np.asarray(pred_support, dtype=float)
    y_support = np.asarray(y_support, dtype=float)
    pred_query = np.asarray(pred_query, dtype=float)
    ok = np.isfinite(pred_support) & np.isfinite(y_support)
    if ok.sum() < 3 or np.nanstd(pred_support[ok]) < 1e-8:
        return residual_calibrate(pred_support, y_support, pred_query)
    a, b = np.polyfit(pred_support[ok], y_support[ok], deg=1)
    return a * pred_query + b


def eval_support_calibration(pred_df, method, encoder, split_file, shots):
    df = pred_df.copy()
    required = {"subject_id", "paired_event_id", "meal_timestamp", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file missing required columns: {missing}")

    df["meal_timestamp"] = pd.to_datetime(df["meal_timestamp"], errors="coerce")
    detail_rows = []

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
            support = sub_all[sub_all["paired_event_id"].isin(support_pairs)]
            query = sub_all[sub_all["paired_event_id"].isin(query_pairs)]
            if len(query) < 3:
                continue

            yq = query["y_true"].to_numpy(dtype=float)
            pq = query["y_pred"].to_numpy(dtype=float)

            m = metrics(yq, pq)
            detail_rows.append({
                "method": method,
                "encoder": encoder,
                "split_file": split_file,
                "fold_subject": subject_id,
                "shot": k,
                "personalization": "global_0shot" if k == 0 else "global_no_update",
                "n_support": len(support),
                "n_query": len(query),
                **m,
            })

            if k > 0 and len(support) >= 1:
                ps = support["y_pred"].to_numpy(dtype=float)
                ys = support["y_true"].to_numpy(dtype=float)

                pq_res = residual_calibrate(ps, ys, pq)
                m = metrics(yq, pq_res)
                detail_rows.append({
                    "method": method,
                    "encoder": encoder,
                    "split_file": split_file,
                    "fold_subject": subject_id,
                    "shot": k,
                    "personalization": "support_residual_calibration",
                    "n_support": len(support),
                    "n_query": len(query),
                    **m,
                })

                pq_aff = affine_calibrate(ps, ys, pq)
                m = metrics(yq, pq_aff)
                detail_rows.append({
                    "method": method,
                    "encoder": encoder,
                    "split_file": split_file,
                    "fold_subject": subject_id,
                    "shot": k,
                    "personalization": "support_affine_calibration",
                    "n_support": len(support),
                    "n_query": len(query),
                    **m,
                })

    return pd.DataFrame(detail_rows)


def summarize(detail):
    group_cols = ["method", "encoder", "split_file", "personalization", "shot"]
    rows = []
    for keys, sub in detail.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_subjects"] = sub["fold_subject"].nunique()
        row["n_query_total"] = int(sub["n_query"].sum())
        for m in ["MAE", "RMSE", "R2", "Pearson"]:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std()
            row[f"{m}_median"] = sub[m].median()
        rows.append(row)
    return pd.DataFrame(rows)


def add_labels(df):
    label_map = {
        "random_meal_split.csv": "Random",
        "cross_subject_split.csv": "Cross-subject",
        "cross_device_dexcom_to_libre.csv": "Dexcom→Libre",
        "cross_device_libre_to_dexcom.csv": "Libre→Dexcom",
        "cross_setting_mealtype_holdout_breakfast.csv": "Breakfast holdout",
    }
    df = df.copy()
    df["split_label"] = df["split_file"].map(label_map).fillna(df["split_file"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/results")
    parser.add_argument("--output_dir", default="outputs/summary")
    parser.add_argument("--pattern", default="*_predictions.csv")
    parser.add_argument("--shots", default="0,1,3,5,10")
    parser.add_argument("--include_unknown", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shots = tuple(int(x.strip()) for x in args.shots.split(",") if x.strip())
    prediction_files = sorted(results_dir.glob(args.pattern))

    all_detail = []
    skipped = []

    for p in prediction_files:
        try:
            small = pd.read_csv(p, nrows=5)
            if not {"y_true", "y_pred"}.issubset(set(small.columns)):
                skipped.append((p.name, "no y_true/y_pred"))
                continue
            full_df = pd.read_csv(p)
            method, encoder, split_file = infer_method_and_split(p.name)
            if method == "Unknown" and not args.include_unknown:
                skipped.append((p.name, "unknown method"))
                continue
            detail = eval_support_calibration(full_df, method, encoder, split_file, shots)
            if not detail.empty:
                all_detail.append(detail)
                print(f"[OK] Processed {p.name}: {len(detail)} rows")
            else:
                skipped.append((p.name, "empty detail"))
        except Exception as e:
            skipped.append((p.name, str(e)))

    if not all_detail:
        raise RuntimeError("No valid prediction files processed.")

    detail_all = pd.concat(all_detail, ignore_index=True)
    summary_all = summarize(detail_all)

    detail_all = add_labels(detail_all)
    summary_all = add_labels(summary_all)

    detail_path = output_dir / "support_calibration_all_detail.csv"
    summary_path = output_dir / "support_calibration_all_summary.csv"
    detail_all.to_csv(detail_path, index=False)
    summary_all.to_csv(summary_path, index=False)

    best = summary_all.loc[summary_all.groupby(["split_file", "shot"])["RMSE_mean"].idxmin()].copy()
    best = best.sort_values(["split_file", "shot"])
    best_path = output_dir / "support_calibration_best_by_split_shot.csv"
    best.to_csv(best_path, index=False)

    best_overall = summary_all.loc[summary_all.groupby(["split_file"])["RMSE_mean"].idxmin()].copy()
    best_overall = best_overall.sort_values("RMSE_mean")
    best_overall_path = output_dir / "support_calibration_best_overall_by_split.csv"
    best_overall.to_csv(best_overall_path, index=False)

    if skipped:
        pd.DataFrame(skipped, columns=["file", "reason"]).to_csv(output_dir / "support_calibration_skipped_files.csv", index=False)

    print(f"[OK] Saved detail: {detail_path}")
    print(f"[OK] Saved summary: {summary_path}")
    print(f"[OK] Saved best by split/shot: {best_path}")
    print(f"[OK] Saved best overall: {best_overall_path}")

    print("\n[BEST OVERALL BY SPLIT]")
    cols = ["split_label", "method", "encoder", "personalization", "shot", "RMSE_mean", "R2_mean", "Pearson_mean"]
    print(best_overall[cols].to_string(index=False))


if __name__ == "__main__":
    main()
