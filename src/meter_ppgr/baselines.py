from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR

from .io_utils import ensure_dir
from .features import load_events, build_tabular_matrix


def _pearson(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(y_true[ok], y_pred[ok])[0, 1])


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": _pearson(y_true, y_pred),
    }


def _models(seed: int):
    models = {
        "Ridge": Ridge(alpha=1.0, random_state=seed),
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed),
        "SVR_RBF": SVR(kernel="rbf", C=10.0, epsilon=0.1),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
    except Exception:
        pass
    return models


def run_tabular_baselines(config: dict) -> None:
    out = Path(config["output_dir"])
    results_dir = ensure_dir(out / "results")
    split_dir = out / "splits"

    meta, seq, rel_grid = load_events(out)
    X, preprocess, numeric_cols, categorical_cols = build_tabular_matrix(meta, seq, rel_grid, config)

    target = config.get("main_target", "iauc_2h")
    if target not in meta.columns:
        raise ValueError(f"Target not found: {target}")

    y = pd.to_numeric(meta[target], errors="coerce")

    split_files = sorted(split_dir.glob("*.csv"))
    if not split_files:
        raise FileNotFoundError("No split files found. Run 03_make_splits.py first.")

    seed = int(config.get("random_seed", 42))
    rows: List[dict] = []

    for split_file in split_files:
        # Avoid exploding runtime for LOSO in the starter baseline; handle separately later.
        if split_file.name == "loso_splits.csv":
            continue

        split = pd.read_csv(split_file)
        split_map = dict(zip(split["event_id"], split["split"]))
        split_label = meta["event_id"].map(split_map).fillna("unused")

        train_mask = (split_label == "train") & y.notna()
        val_mask = (split_label == "val") & y.notna()
        test_mask = (split_label == "test") & y.notna()

        if train_mask.sum() < 10 or test_mask.sum() < 5:
            continue

        for model_name, model in _models(seed).items():
            pipe = Pipeline([
                ("preprocess", preprocess),
                ("model", model),
            ])
            pipe.fit(X.loc[train_mask], y.loc[train_mask])
            pred = pipe.predict(X.loc[test_mask])
            m = regression_metrics(y.loc[test_mask], pred)
            row = {
                "split_file": split_file.name,
                "model": model_name,
                "target": target,
                "n_train": int(train_mask.sum()),
                "n_val": int(val_mask.sum()),
                "n_test": int(test_mask.sum()),
                **m,
            }
            rows.append(row)
            print(f"[{split_file.name}][{model_name}] RMSE={m['RMSE']:.3f} MAE={m['MAE']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(results_dir / f"tabular_baselines_{target}.csv", index=False)
    print(f"[OK] Saved baseline results to {results_dir / f'tabular_baselines_{target}.csv'}")
