import argparse
from pathlib import Path
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


def eval_file(pred_df, method, encoder, split_file, shots):
    df = pred_df.copy()
    required = {"subject_id", "paired_event_id", "meal_timestamp", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

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
            support = sub_all[sub_all["paired_event_id"].isin(support_pairs)]
            query = sub_all[sub_all["paired_event_id"].isin(query_pairs)]
            if len(query) < 3:
                continue

            yq = query["y_true"].to_numpy(dtype=float)
            pq = query["y_pred"].to_numpy(dtype=float)

            m = metrics(yq, pq)
            rows.append({
                "method": method,
                "encoder": encoder,
                "split_file": split_file,
                "split_label": LABEL_MAP.get(split_file, split_file),
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
                rows.append({
                    "method": method,
                    "encoder": encoder,
                    "split_file": split_file,
                    "split_label": LABEL_MAP.get(split_file, split_file),
                    "fold_subject": subject_id,
                    "shot": k,
                    "personalization": "support_residual_calibration",
                    "n_support": len(support),
                    "n_query": len(query),
                    **m,
                })

                pq_aff = affine_calibrate(ps, ys, pq)
                m = metrics(yq, pq_aff)
                rows.append({
                    "method": method,
                    "encoder": encoder,
                    "split_file": split_file,
                    "split_label": LABEL_MAP.get(split_file, split_file),
                    "fold_subject": subject_id,
                    "shot": k,
                    "personalization": "support_affine_calibration",
                    "n_support": len(support),
                    "n_query": len(query),
                    **m,
                })

    return pd.DataFrame(rows)


def summarize(detail):
    group_cols = ["method", "encoder", "split_file", "split_label", "personalization", "shot"]
    rows = []
    for keys, sub in detail.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_subjects"] = sub["fold_subject"].nunique()
        row["n_query_total"] = int(sub["n_query"].sum())
        for metric in ["MAE", "RMSE", "R2", "Pearson"]:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_std"] = sub[metric].std()
            row[f"{metric}_median"] = sub[metric].median()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/results")
    parser.add_argument("--output_dir", default="outputs/supplement_exp2")
    parser.add_argument("--pattern", default="*_predictions.csv")
    parser.add_argument("--shots", default="0,1,3,5,10")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shots = tuple(int(x.strip()) for x in args.shots.split(",") if x.strip())

    files = sorted(results_dir.glob(args.pattern))
    detail_frames = []
    skipped = []

    for p in files:
        try:
            small = pd.read_csv(p, nrows=5)
            if not {"y_true", "y_pred"}.issubset(set(small.columns)):
                skipped.append((p.name, "missing y_true/y_pred"))
                continue
            method, encoder, split_file = infer_method_and_split(p.name)
            if method == "Unknown":
                skipped.append((p.name, "unknown pattern"))
                continue
            df = pd.read_csv(p)
            detail = eval_file(df, method, encoder, split_file, shots)
            if detail.empty:
                skipped.append((p.name, "empty detail"))
                continue
            detail["source_file"] = p.name
            detail_frames.append(detail)
            print(f"[OK] {p.name}: {len(detail)} detail rows")
        except Exception as e:
            skipped.append((p.name, str(e)))

    if not detail_frames:
        raise RuntimeError("No valid prediction files processed.")

    detail_all = pd.concat(detail_frames, ignore_index=True)
    summary_all = summarize(detail_all)

    detail_path = output_dir / "supplement_exp2_support_all_detail.csv"
    summary_path = output_dir / "supplement_exp2_support_all_summary.csv"
    detail_all.to_csv(detail_path, index=False)
    summary_all.to_csv(summary_path, index=False)

    best_by_split_shot = summary_all.loc[summary_all.groupby(["split_file", "shot"])["RMSE_mean"].idxmin()].copy()
    best_by_split_shot = best_by_split_shot.sort_values(["split_file", "shot"])
    best_by_split_shot.to_csv(output_dir / "supplement_exp2_best_by_split_shot.csv", index=False)

    best_overall = summary_all.loc[summary_all.groupby("split_file")["RMSE_mean"].idxmin()].copy()
    best_overall = best_overall.sort_values("RMSE_mean")
    best_overall.to_csv(output_dir / "supplement_exp2_best_overall_by_split.csv", index=False)

    # Strong tabular vs METER-PSC fixed comparison.
    tabular = summary_all[summary_all["method"].str.startswith("Tabular-", na=False)].copy()
    meter_psc = summary_all[
        (summary_all["method"].eq("METER-v1")) &
        (summary_all["encoder"].eq("tcn")) &
        (summary_all["personalization"].eq("support_residual_calibration")) &
        (summary_all["shot"].eq(5))
    ].copy()

    comp_rows = []
    for split_file in sorted(summary_all["split_file"].unique()):
        tab_split = tabular[tabular["split_file"].eq(split_file)]
        psc_split = meter_psc[meter_psc["split_file"].eq(split_file)]
        if tab_split.empty or psc_split.empty:
            continue
        best_tab = tab_split.loc[tab_split["RMSE_mean"].idxmin()]
        psc = psc_split.iloc[0]
        comp_rows.append({
            "split_file": split_file,
            "split_label": LABEL_MAP.get(split_file, split_file),
            "best_tabular_support_method": best_tab["method"],
            "best_tabular_support_personalization": best_tab["personalization"],
            "best_tabular_support_shot": best_tab["shot"],
            "best_tabular_support_RMSE": best_tab["RMSE_mean"],
            "best_tabular_support_MAE": best_tab["MAE_mean"],
            "best_tabular_support_R2": best_tab["R2_mean"],
            "best_tabular_support_Pearson": best_tab["Pearson_mean"],
            "METER_PSC_method": "METER-v1-TCN + 5-shot residual",
            "METER_PSC_RMSE": psc["RMSE_mean"],
            "METER_PSC_MAE": psc["MAE_mean"],
            "METER_PSC_R2": psc["R2_mean"],
            "METER_PSC_Pearson": psc["Pearson_mean"],
            "METER_PSC_improvement_vs_best_tabular_support_%": (best_tab["RMSE_mean"] - psc["RMSE_mean"]) / best_tab["RMSE_mean"] * 100,
        })

    if comp_rows:
        pd.DataFrame(comp_rows).to_csv(output_dir / "supplement_exp2_meter_psc_vs_best_tabular_support.csv", index=False)

    if skipped:
        pd.DataFrame(skipped, columns=["file", "reason"]).to_csv(output_dir / "supplement_exp2_skipped_files.csv", index=False)

    print(f"[OK] Saved detail: {detail_path}")
    print(f"[OK] Saved summary: {summary_path}")
    print(f"[OK] Saved best overall: {output_dir / 'supplement_exp2_best_overall_by_split.csv'}")
    if comp_rows:
        print(f"[OK] Saved METER-PSC vs best tabular support comparison.")
        print(pd.DataFrame(comp_rows).to_string(index=False))


if __name__ == "__main__":
    main()
