from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from .features import load_events, build_tabular_matrix
from .stage2_utils import ensure_dir, regression_metrics, summarize_by_group


def _get_models(seed: int) -> Dict[str, object]:
    models = {
        "Ridge": Ridge(alpha=1.0, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, random_state=seed),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, objective="reg:squarederror", random_state=seed, n_jobs=-1)
    except Exception:
        pass
    return models


def _subject_pair_order(meta: pd.DataFrame, subject_id: str) -> List[str]:
    sub = meta[meta["subject_id"] == subject_id].copy()
    sub["meal_timestamp"] = pd.to_datetime(sub["meal_timestamp"], errors="coerce")
    return sub.groupby("paired_event_id")["meal_timestamp"].min().sort_values().index.tolist()


def _residual_calibrate(pred_support, y_support, pred_query):
    if len(y_support) == 0:
        return pred_query
    return np.asarray(pred_query) + float(np.nanmean(np.asarray(y_support) - np.asarray(pred_support)))


def _affine_calibrate(pred_support, y_support, pred_query):
    if len(y_support) < 3:
        return _residual_calibrate(pred_support, y_support, pred_query)
    x = np.asarray(pred_support, dtype=float)
    y = np.asarray(y_support, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.nanstd(x[ok]) < 1e-8:
        return _residual_calibrate(pred_support, y_support, pred_query)
    a, b = np.polyfit(x[ok], y[ok], deg=1)
    return a * np.asarray(pred_query) + b


def run_cold_start_tabular(config: dict, shots=(0, 1, 3, 5, 10)) -> None:
    out = Path(config["output_dir"])
    res_dir = ensure_dir(out / "results")
    meta, seq, rel_grid = load_events(out)
    X, preprocess, _, _ = build_tabular_matrix(meta, seq, rel_grid, config)
    target = config.get("main_target", "iauc_2h")
    y = pd.to_numeric(meta[target], errors="coerce")
    subjects = sorted(meta["subject_id"].dropna().unique().tolist())
    seed = int(config.get("random_seed", 42))
    rows = []
    for model_name, model in _get_models(seed).items():
        for s in subjects:
            train_base_mask = (meta["subject_id"] != s) & y.notna()
            subject_mask = (meta["subject_id"] == s) & y.notna()
            pair_order = _subject_pair_order(meta, s)
            if train_base_mask.sum() < 20 or subject_mask.sum() < 6 or len(pair_order) < 4:
                continue
            pipe = Pipeline([("preprocess", preprocess), ("model", model)])
            pipe.fit(X.loc[train_base_mask], y.loc[train_base_mask])
            sub_idx = meta.index[subject_mask].to_numpy()
            sub_pred = pd.Series(pipe.predict(X.loc[subject_mask]), index=sub_idx)
            for k in shots:
                if len(pair_order) <= k + 1:
                    continue
                support_pairs = set(pair_order[:k])
                query_pairs = set(pair_order[k:])
                support_mask = subject_mask & meta["paired_event_id"].isin(support_pairs)
                query_mask = subject_mask & meta["paired_event_id"].isin(query_pairs)
                if query_mask.sum() < 3:
                    continue
                pred_query_global = sub_pred.loc[meta.index[query_mask]].to_numpy()
                y_query = y.loc[query_mask].to_numpy()
                m = regression_metrics(y_query, pred_query_global)
                rows.append({"model": model_name, "personalization": "global_0shot" if k == 0 else "global_no_update", "fold_subject": s, "shot": k, "n_support": int(support_mask.sum()), "n_query": int(query_mask.sum()), **m})
                if k > 0 and support_mask.sum() >= 1:
                    pred_support = sub_pred.loc[meta.index[support_mask]].to_numpy()
                    y_support = y.loc[support_mask].to_numpy()
                    pred_resid = _residual_calibrate(pred_support, y_support, pred_query_global)
                    m = regression_metrics(y_query, pred_resid)
                    rows.append({"model": model_name, "personalization": "support_residual_calibration", "fold_subject": s, "shot": k, "n_support": int(support_mask.sum()), "n_query": int(query_mask.sum()), **m})
                    pred_affine = _affine_calibrate(pred_support, y_support, pred_query_global)
                    m = regression_metrics(y_query, pred_affine)
                    rows.append({"model": model_name, "personalization": "support_affine_calibration", "fold_subject": s, "shot": k, "n_support": int(support_mask.sum()), "n_query": int(query_mask.sum()), **m})
            print(f"[COLD START][{model_name}][{s}] done")
    df = pd.DataFrame(rows)
    detail_path = res_dir / f"cold_start_tabular_detail_{target}.csv"
    df.to_csv(detail_path, index=False)
    summary = summarize_by_group(df, ["model", "personalization", "shot"])
    summary_path = res_dir / f"cold_start_tabular_summary_{target}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Saved cold-start detail: {detail_path}")
    print(f"[OK] Saved cold-start summary: {summary_path}")
