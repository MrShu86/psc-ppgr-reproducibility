from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from .features import load_events, build_tabular_matrix
from .stage2_utils import ensure_dir, regression_metrics, summarize_by_group


def get_models(seed: int, fast: bool = True) -> Dict[str, object]:
    models = {
        "Ridge": Ridge(alpha=1.0, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, random_state=seed),
    }
    if not fast:
        models["RandomForest"] = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1)
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, objective="reg:squarederror", random_state=seed, n_jobs=-1)
        except Exception:
            pass
    return models


def run_loso_tabular(config: dict, fast: bool = True) -> None:
    out = Path(config["output_dir"])
    res_dir = ensure_dir(out / "results")
    meta, seq, rel_grid = load_events(out)
    X, preprocess, _, _ = build_tabular_matrix(meta, seq, rel_grid, config)
    target = config.get("main_target", "iauc_2h")
    y = pd.to_numeric(meta[target], errors="coerce")
    subjects = sorted(meta["subject_id"].dropna().unique().tolist())
    seed = int(config.get("random_seed", 42))
    rows = []
    for model_name, model in get_models(seed, fast=fast).items():
        for s in subjects:
            train_mask = (meta["subject_id"] != s) & y.notna()
            test_mask = (meta["subject_id"] == s) & y.notna()
            if train_mask.sum() < 20 or test_mask.sum() < 3:
                continue
            pipe = Pipeline([("preprocess", preprocess), ("model", model)])
            pipe.fit(X.loc[train_mask], y.loc[train_mask])
            pred = pipe.predict(X.loc[test_mask])
            m = regression_metrics(y.loc[test_mask], pred)
            rows.append({"model": model_name, "fold_subject": s, "target": target, "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()), **m})
            print(f"[LOSO][{model_name}][{s}] n={test_mask.sum()} RMSE={m['RMSE']:.2f} R2={m['R2']:.3f}")
    df = pd.DataFrame(rows)
    detail_path = res_dir / f"loso_tabular_detail_{target}.csv"
    df.to_csv(detail_path, index=False)
    summary = summarize_by_group(df, ["model"])
    summary_path = res_dir / f"loso_tabular_summary_{target}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Saved LOSO detail: {detail_path}")
    print(f"[OK] Saved LOSO summary: {summary_path}")
