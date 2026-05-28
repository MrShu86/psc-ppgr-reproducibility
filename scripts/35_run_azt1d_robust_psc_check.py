import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
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
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Pearson": np.nan, "bias": np.nan}
    err = y_pred - y_true
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "Pearson": pearson(y_true, y_pred),
        "bias": float(np.mean(err)),
        "abs_bias": float(abs(np.mean(err))),
        "p90_abs_error": float(np.quantile(np.abs(err), 0.90)),
    }


def get_models(seed, model_names):
    all_models = {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.25, random_state=seed, max_iter=10000),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed),
        "RandomForest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=seed, n_jobs=-1),
    }
    try:
        from xgboost import XGBRegressor
        all_models["XGBoost"] = XGBRegressor(
            n_estimators=600,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
    except Exception as e:
        print(f"[WARN] XGBoost unavailable: {e}")

    if model_names == ["all"]:
        return all_models

    out = {}
    for name in model_names:
        if name in all_models:
            out[name] = all_models[name]
        else:
            print(f"[WARN] Unknown or unavailable model: {name}")
    if not out:
        raise RuntimeError("No valid model selected.")
    return out


def make_preprocess(X):
    numeric_cols, categorical_cols = [], []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])
    return pre, numeric_cols, categorical_cols


def add_baseline_bins(events, train_mask=None):
    events = events.copy()
    if "baseline_glucose" not in events.columns:
        events["baseline_bin"] = "unknown"
        return events

    s = pd.to_numeric(events["baseline_glucose"], errors="coerce")
    if train_mask is None:
        ref = s.dropna()
    else:
        ref = s.loc[train_mask].dropna()

    if len(ref) < 10:
        events["baseline_bin"] = "unknown"
        return events

    q1, q2 = ref.quantile([1/3, 2/3]).tolist()

    def bin_one(x):
        if not np.isfinite(x):
            return "unknown"
        if x <= q1:
            return "low"
        if x <= q2:
            return "mid"
        return "high"

    events["baseline_bin"] = s.map(bin_one)
    return events


def build_features(events, target):
    drop_cols = {
        "event_id", "subject_id", "meal_time", "source_file",
        "iauc_pos_2h", "auc_delta_2h", "post_mean_delta_2h", "post_max_delta_2h", "post_min_delta_2h",
        "y_true", "y_pred", "y_pred_calibrated"
    }
    feature_cols = [c for c in events.columns if c not in drop_cols]
    X = events[feature_cols].copy()
    y = pd.to_numeric(events[target], errors="coerce")
    return X, y, feature_cols


def support_delta(support, method, query_row=None, tau=5.0, clip=None, min_match=2):
    """
    Return delta = y_true - y_pred estimated from support.
    method controls robust or matched variants.
    """
    if support.empty:
        return 0.0, 0, "empty_support"

    sup = support.copy()

    # matched support subsets
    if method.startswith("mealtype_") and query_row is not None and "meal_type" in sup.columns:
        matched = sup[sup["meal_type"].astype(str).eq(str(query_row.get("meal_type", "")))]
        if len(matched) >= min_match:
            sup = matched
            match_status = "mealtype_matched"
        else:
            match_status = "mealtype_fallback_global"
    elif method.startswith("baselinebin_") and query_row is not None and "baseline_bin" in sup.columns:
        matched = sup[sup["baseline_bin"].astype(str).eq(str(query_row.get("baseline_bin", "")))]
        if len(matched) >= min_match:
            sup = matched
            match_status = "baselinebin_matched"
        else:
            match_status = "baselinebin_fallback_global"
    else:
        match_status = "global"

    residual = pd.to_numeric(sup["y_true"], errors="coerce") - pd.to_numeric(sup["y_pred"], errors="coerce")
    residual = residual.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    n = len(residual)
    if n == 0:
        return 0.0, 0, "no_valid_residual"

    base_method = method
    for prefix in ["mealtype_", "baselinebin_"]:
        if base_method.startswith(prefix):
            base_method = base_method.replace(prefix, "", 1)

    if base_method.startswith("mean"):
        delta = float(np.mean(residual))
    elif base_method.startswith("median"):
        delta = float(np.median(residual))
    elif base_method.startswith("trimmed"):
        if n >= 5:
            lo, hi = np.quantile(residual, [0.1, 0.9])
            residual2 = residual[(residual >= lo) & (residual <= hi)]
            delta = float(np.mean(residual2)) if len(residual2) else float(np.mean(residual))
        else:
            delta = float(np.median(residual))
    else:
        delta = float(np.mean(residual))

    # shrinkage variants
    if "shrink" in base_method:
        delta = float(n / (n + tau) * delta)

    if clip is not None:
        delta = float(np.clip(delta, -clip, clip))

    return delta, n, match_status


def calibrate_query(support, query, calibration, tau=5.0, clip=None, min_match=2):
    pred = []
    delta_list = []
    n_match_list = []
    match_status_list = []

    if calibration == "global_no_update":
        return query["y_pred"].to_numpy(dtype=float), [np.nan] * len(query), [0] * len(query), ["no_update"] * len(query)

    if calibration == "affine":
        ps = support["y_pred"].to_numpy(dtype=float)
        ys = support["y_true"].to_numpy(dtype=float)
        ok = np.isfinite(ps) & np.isfinite(ys)
        if ok.sum() >= 3 and np.nanstd(ps[ok]) > 1e-8:
            a, b = np.polyfit(ps[ok], ys[ok], 1)
            return a * query["y_pred"].to_numpy(dtype=float) + b, [b] * len(query), [ok.sum()] * len(query), ["affine"] * len(query)
        else:
            calibration = "mean_shrink"

    for _, q in query.iterrows():
        delta, n_match, match_status = support_delta(
            support,
            method=calibration,
            query_row=q,
            tau=tau,
            clip=clip,
            min_match=min_match,
        )
        pred.append(float(q["y_pred"] + delta))
        delta_list.append(delta)
        n_match_list.append(n_match)
        match_status_list.append(match_status)

    return np.asarray(pred, dtype=float), delta_list, n_match_list, match_status_list


def evaluate(events, target, models, shots, calibrations, min_query, tau_values, clip_values, min_match):
    events = events.copy()
    events["meal_time"] = pd.to_datetime(events["meal_time"], errors="coerce")
    events = events.dropna(subset=["meal_time", target, "subject_id"]).sort_values(["subject_id", "meal_time"])
    events["subject_id"] = events["subject_id"].astype(str)

    # Initial baseline bins from all events; in each LOSO model we recompute train-based bins.
    subjects = sorted(events["subject_id"].unique().tolist())

    rows_detail = []
    rows_pred = []

    for model_name, model in models.items():
        for test_subject in subjects:
            base_test_mask = events["subject_id"].eq(test_subject)
            base_train_mask = ~base_test_mask
            if base_train_mask.sum() < 20:
                continue

            # Train-based baseline bins to avoid using test distribution thresholds.
            ev = add_baseline_bins(events, train_mask=base_train_mask)
            X_all, y_all, feature_cols = build_features(ev, target)
            pre, numeric_cols, categorical_cols = make_preprocess(X_all)

            test_mask = ev["subject_id"].eq(test_subject)
            train_mask = ~test_mask

            if test_mask.sum() < max(shots) + min_query:
                continue

            pipe = Pipeline([
                ("preprocess", pre),
                ("model", model),
            ])
            pipe.fit(X_all.loc[train_mask], y_all.loc[train_mask])

            test_events = ev.loc[test_mask].copy().sort_values("meal_time")
            pred_test = pipe.predict(X_all.loc[test_mask])
            test_events["y_true"] = y_all.loc[test_mask].to_numpy(dtype=float)
            test_events["y_pred"] = pred_test

            for k in shots:
                if len(test_events) <= k + min_query:
                    continue

                support = test_events.iloc[:k].copy()
                query = test_events.iloc[k:].copy()
                if len(query) < min_query:
                    continue

                # Always evaluate no-update query on the same query subset.
                yq = query["y_true"].to_numpy(dtype=float)
                pq = query["y_pred"].to_numpy(dtype=float)
                m = metrics(yq, pq)

                rows_detail.append({
                    "dataset": "AZT1D",
                    "setting": "external_loso_cross_subject",
                    "model": model_name,
                    "subject_id": test_subject,
                    "shot": k,
                    "personalization": "global_no_update" if k > 0 else "global_0shot",
                    "tau": np.nan,
                    "clip": np.nan,
                    "n_support": len(support),
                    "n_query": len(query),
                    **m,
                })

                q0 = query.copy()
                q0["dataset"] = "AZT1D"
                q0["setting"] = "external_loso_cross_subject"
                q0["model"] = model_name
                q0["shot"] = k
                q0["personalization"] = "global_no_update" if k > 0 else "global_0shot"
                q0["tau"] = np.nan
                q0["clip"] = np.nan
                q0["y_pred_calibrated"] = q0["y_pred"]
                q0["support_delta"] = np.nan
                q0["n_matched_support"] = 0
                q0["match_status"] = "no_update"
                rows_pred.append(q0)

                if k == 0:
                    continue

                for cal in calibrations:
                    # expanded variants with tau/clip settings
                    variants = []
                    if cal.endswith("_shrink"):
                        for tau in tau_values:
                            variants.append((cal, tau, None, f"{cal}_tau{tau:g}"))
                    elif cal.endswith("_clipped"):
                        for clip in clip_values:
                            variants.append((cal.replace("_clipped", ""), np.nan, clip, f"{cal}_clip{clip:g}"))
                    else:
                        variants.append((cal, np.nan, None, cal))

                    for base_cal, tau, clip, label in variants:
                        tau_eff = float(tau) if np.isfinite(tau) else 5.0
                        pred_cal, deltas, n_match, match_status = calibrate_query(
                            support, query, base_cal, tau=tau_eff, clip=clip, min_match=min_match
                        )
                        m = metrics(yq, pred_cal)

                        rows_detail.append({
                            "dataset": "AZT1D",
                            "setting": "external_loso_cross_subject",
                            "model": model_name,
                            "subject_id": test_subject,
                            "shot": k,
                            "personalization": label,
                            "tau": tau if np.isfinite(tau) else np.nan,
                            "clip": clip if clip is not None else np.nan,
                            "n_support": len(support),
                            "n_query": len(query),
                            "mean_matched_support": float(np.mean(n_match)) if len(n_match) else np.nan,
                            **m,
                        })

                        qc = query.copy()
                        qc["dataset"] = "AZT1D"
                        qc["setting"] = "external_loso_cross_subject"
                        qc["model"] = model_name
                        qc["shot"] = k
                        qc["personalization"] = label
                        qc["tau"] = tau if np.isfinite(tau) else np.nan
                        qc["clip"] = clip if clip is not None else np.nan
                        qc["y_pred_calibrated"] = pred_cal
                        qc["support_delta"] = deltas
                        qc["n_matched_support"] = n_match
                        qc["match_status"] = match_status
                        rows_pred.append(qc)

    detail = pd.DataFrame(rows_detail)
    pred = pd.concat(rows_pred, ignore_index=True) if rows_pred else pd.DataFrame()
    return detail, pred


def summarize(detail):
    rows = []
    for keys, sub in detail.groupby(["dataset", "setting", "model", "personalization", "shot"], dropna=False):
        dataset, setting, model, pers, shot = keys
        row = {
            "dataset": dataset,
            "setting": setting,
            "model": model,
            "personalization": pers,
            "shot": shot,
            "n_subjects": sub["subject_id"].nunique(),
            "n_query_total": int(sub["n_query"].sum()),
        }
        for metric in ["MAE", "RMSE", "R2", "Pearson", "bias", "abs_bias", "p90_abs_error"]:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_std"] = sub[metric].std()
            row[f"{metric}_median"] = sub[metric].median()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "shot", "RMSE_mean"])


def paired_tests(detail):
    try:
        from scipy import stats
    except Exception:
        stats = None

    rows = []
    for keys, sub in detail.groupby(["dataset", "setting", "model", "shot"], dropna=False):
        dataset, setting, model, shot = keys
        if int(shot) == 0:
            continue
        base = sub[sub["personalization"].eq("global_no_update")]
        if base.empty:
            continue

        for pers in sorted([x for x in sub["personalization"].unique() if x not in ["global_no_update", "global_0shot"]]):
            cal = sub[sub["personalization"].eq(pers)]
            if cal.empty:
                continue
            merged = base[["subject_id", "RMSE", "MAE", "bias"]].merge(
                cal[["subject_id", "RMSE", "MAE", "bias"]],
                on="subject_id",
                suffixes=("_no_update", "_cal")
            )
            if len(merged) < 3:
                continue

            diff = merged["RMSE_no_update"].to_numpy(dtype=float) - merged["RMSE_cal"].to_numpy(dtype=float)
            row = {
                "dataset": dataset,
                "setting": setting,
                "model": model,
                "shot": shot,
                "personalization": pers,
                "n_subjects": len(merged),
                "no_update_RMSE_mean": float(merged["RMSE_no_update"].mean()),
                "calibrated_RMSE_mean": float(merged["RMSE_cal"].mean()),
                "mean_delta_RMSE": float(np.mean(diff)),
                "median_delta_RMSE": float(np.median(diff)),
                "relative_RMSE_reduction_%": float((merged["RMSE_no_update"].mean() - merged["RMSE_cal"].mean()) / merged["RMSE_no_update"].mean() * 100),
                "improvement_rate_%": float(np.mean(diff > 0) * 100),
                "wilcoxon_p": np.nan,
                "paired_t_p": np.nan,
            }
            if stats is not None and np.any(diff != 0):
                try:
                    row["wilcoxon_p"] = float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
                except Exception:
                    pass
                try:
                    row["paired_t_p"] = float(stats.ttest_1samp(diff, 0.0).pvalue)
                except Exception:
                    pass
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "shot", "relative_RMSE_reduction_%"], ascending=[True, True, False])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="outputs_azt1d/azt1d_meal_events.csv")
    parser.add_argument("--output_dir", default="outputs_azt1d/robust_psc_check")
    parser.add_argument("--target", default="iauc_pos_2h")
    parser.add_argument("--models", default="HistGradientBoosting,XGBoost,RandomForest,Ridge")
    parser.add_argument("--shots", default="0,1,3,5,10,20,30")
    parser.add_argument(
        "--calibrations",
        default="mean,median,trimmed,mean_shrink,median_shrink,mean_clipped,mealtype_mean,mealtype_mean_shrink,baselinebin_mean,baselinebin_mean_shrink,affine",
        help="Comma-separated robust PSC variants."
    )
    parser.add_argument("--tau_values", default="5,10,20")
    parser.add_argument("--clip_values", default="500,1000,1500")
    parser.add_argument("--min_query", type=int, default=3)
    parser.add_argument("--min_match", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    if args.target not in events.columns:
        raise ValueError(f"Target {args.target} not in events.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = [x.strip() for x in args.models.split(",") if x.strip()]
    models = get_models(args.seed, model_names)

    shots = tuple(int(x.strip()) for x in args.shots.split(",") if x.strip())
    calibrations = [x.strip() for x in args.calibrations.split(",") if x.strip()]
    tau_values = tuple(float(x.strip()) for x in args.tau_values.split(",") if x.strip())
    clip_values = tuple(float(x.strip()) for x in args.clip_values.split(",") if x.strip())

    print("[INFO] Running AZT1D robust PSC sensitivity check...")
    print(f"[INFO] events={len(events)}, subjects={events['subject_id'].nunique()}, models={list(models.keys())}")
    print(f"[INFO] shots={shots}")
    print(f"[INFO] calibrations={calibrations}")

    detail, pred = evaluate(events, args.target, models, shots, calibrations, args.min_query, tau_values, clip_values, args.min_match)
    if detail.empty:
        raise RuntimeError("No evaluation rows generated.")

    summary = summarize(detail)
    tests = paired_tests(detail)

    detail.to_csv(output_dir / "azt1d_robust_psc_subject_detail.csv", index=False)
    pred.to_csv(output_dir / "azt1d_robust_psc_predictions.csv", index=False)
    summary.to_csv(output_dir / "azt1d_robust_psc_summary.csv", index=False)
    tests.to_csv(output_dir / "azt1d_robust_psc_paired_tests.csv", index=False)

    # Best calibrated method per shot and model, and global best by shot.
    calibrated = summary[~summary["personalization"].isin(["global_no_update", "global_0shot"])].copy()
    if not calibrated.empty:
        best_by_model_shot = calibrated.loc[calibrated.groupby(["model", "shot"])["RMSE_mean"].idxmin()].copy()
        best_by_model_shot.to_csv(output_dir / "azt1d_robust_psc_best_by_model_shot.csv", index=False)

        best_by_shot = summary.loc[summary.groupby(["shot"])["RMSE_mean"].idxmin()].copy()
        best_by_shot.to_csv(output_dir / "azt1d_robust_psc_best_by_shot.csv", index=False)

    # Compare best calibrated vs no-update for each model/shot.
    rows = []
    for (model, shot), sub in summary.groupby(["model", "shot"]):
        base = sub[sub["personalization"].isin(["global_no_update", "global_0shot"])]
        cal = sub[~sub["personalization"].isin(["global_no_update", "global_0shot"])]
        if base.empty or cal.empty or int(shot) == 0:
            continue
        best_cal = cal.loc[cal["RMSE_mean"].idxmin()]
        base_row = base.iloc[0]
        rows.append({
            "model": model,
            "shot": shot,
            "no_update_RMSE": base_row["RMSE_mean"],
            "best_calibration": best_cal["personalization"],
            "best_calibrated_RMSE": best_cal["RMSE_mean"],
            "best_calibrated_MAE": best_cal["MAE_mean"],
            "best_calibrated_bias": best_cal["bias_mean"],
            "relative_RMSE_reduction_%": (base_row["RMSE_mean"] - best_cal["RMSE_mean"]) / base_row["RMSE_mean"] * 100,
        })
    pd.DataFrame(rows).to_csv(output_dir / "azt1d_robust_psc_best_calibrated_vs_no_update.csv", index=False)

    print(f"[OK] Saved robust PSC results to {output_dir}")
    print("\n[BEST CALIBRATED VS NO-UPDATE]")
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    print("\n[TOP PAIRED TESTS]")
    if not tests.empty:
        print(tests.sort_values("relative_RMSE_reduction_%", ascending=False).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
