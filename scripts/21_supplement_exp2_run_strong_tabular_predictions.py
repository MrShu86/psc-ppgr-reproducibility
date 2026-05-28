import argparse
from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meter_ppgr.io_utils import load_config
from meter_ppgr.features import load_events, build_tabular_matrix


DEFAULT_SPLITS = [
    "random_meal_split.csv",
    "cross_device_dexcom_to_libre.csv",
    "cross_device_libre_to_dexcom.csv",
    "cross_subject_split.csv",
    "cross_setting_mealtype_holdout_breakfast.csv",
]


def safe_name(s: str) -> str:
    s = s.replace(".csv", "")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    return s


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
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "Pearson": pearson(y_true, y_pred),
    }


def get_models(seed: int, model_names):
    all_models = {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.25, random_state=seed, max_iter=10000),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=seed
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=3, random_state=seed, n_jobs=-1
        ),
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

    try:
        from lightgbm import LGBMRegressor
        all_models["LightGBM"] = LGBMRegressor(
            n_estimators=600,
            max_depth=-1,
            learning_rate=0.03,
            subsample=0.90,
            colsample_bytree=0.90,
            random_state=seed,
            n_jobs=-1,
        )
    except Exception as e:
        print(f"[WARN] LightGBM unavailable: {e}")

    try:
        from catboost import CatBoostRegressor
        all_models["CatBoost"] = CatBoostRegressor(
            iterations=600,
            depth=5,
            learning_rate=0.03,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
        )
    except Exception as e:
        print(f"[WARN] CatBoost unavailable: {e}")

    if model_names == ["all"]:
        return all_models

    out = {}
    for name in model_names:
        if name in all_models:
            out[name] = all_models[name]
        else:
            print(f"[WARN] Requested model not available or unknown: {name}")
    if not out:
        raise RuntimeError("No valid models selected.")
    return out


def run_strong_tabular_predictions(config, splits, model_names):
    out = Path(config["output_dir"])
    results_dir = out / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    meta, seq, rel_grid = load_events(out)
    X, preprocess, numeric_cols, categorical_cols = build_tabular_matrix(meta, seq, rel_grid, config)

    target = config.get("main_target", "iauc_2h")
    y = pd.to_numeric(meta[target], errors="coerce")
    seed = int(config.get("random_seed", 42))

    models = get_models(seed, model_names)
    all_rows = []

    for split_name in splits:
        split_path = out / "splits" / split_name
        if not split_path.exists():
            print(f"[WARN] Missing split file: {split_path}")
            continue

        split = pd.read_csv(split_path)
        split_map = dict(zip(split["event_id"], split["split"]))
        split_label = meta["event_id"].map(split_map).fillna("unused")

        train_mask = (split_label == "train") & y.notna()
        val_mask = (split_label == "val") & y.notna()
        test_mask = (split_label == "test") & y.notna()

        if train_mask.sum() < 20 or test_mask.sum() < 5:
            print(f"[WARN] Too few samples for {split_name}: train={train_mask.sum()}, test={test_mask.sum()}")
            continue

        for model_name, model in models.items():
            pipe = Pipeline([
                ("preprocess", preprocess),
                ("model", model),
            ])

            print(f"[RUN][{model_name}][{split_name}] train={train_mask.sum()} test={test_mask.sum()}")
            pipe.fit(X.loc[train_mask], y.loc[train_mask])
            pred = pipe.predict(X.loc[test_mask])
            y_true = y.loc[test_mask].to_numpy(dtype=float)
            m = metrics(y_true, pred)

            row = {
                "split_file": split_name,
                "model": model_name,
                "target": target,
                "n_train": int(train_mask.sum()),
                "n_val": int(val_mask.sum()),
                "n_test": int(test_mask.sum()),
                "feature_mode": config.get("feature_mode", "causal_premeal"),
                **m,
            }
            all_rows.append(row)
            print(f"[RESULT][{model_name}][{split_name}] RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}, R2={m['R2']:.3f}, r={m['Pearson']:.3f}")

            # Save prediction file for support calibration.
            pred_df = meta.loc[test_mask, [
                "event_id", "paired_event_id", "subject_id", "device", "meal_type", "meal_timestamp"
            ]].copy()
            pred_df["y_true"] = y_true
            pred_df["y_pred"] = pred
            pred_df["abs_error"] = np.abs(y_true - pred)

            pred_path = results_dir / f"tabular_baseline_{safe_name(model_name)}_{safe_name(split_name)}_{target}_predictions.csv"
            pred_df.to_csv(pred_path, index=False)

            res_path = results_dir / f"tabular_baseline_{safe_name(model_name)}_{safe_name(split_name)}_{target}.csv"
            pd.DataFrame([row]).to_csv(res_path, index=False)

    if all_rows:
        summary = pd.DataFrame(all_rows)
        summary_path = results_dir / f"strong_tabular_prediction_results_{target}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[OK] Saved strong tabular result summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument(
        "--models",
        default="Ridge,ElasticNet,HistGradientBoosting,RandomForest,XGBoost",
        help="Comma-separated model names, or 'all'. Optional if installed: LightGBM, CatBoost."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    model_names = [x.strip() for x in args.models.split(",") if x.strip()]
    run_strong_tabular_predictions(config, splits, model_names)


if __name__ == "__main__":
    main()
