import argparse
from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meter_ppgr.io_utils import load_config
from meter_ppgr.features import load_events, build_tabular_matrix


DEFAULT_SPLITS = [
    "cross_device_dexcom_to_libre.csv",
    "cross_device_libre_to_dexcom.csv",
    "cross_subject_split.csv",
    "cross_setting_mealtype_holdout_breakfast.csv",
]

LABEL_MAP = {
    "random_meal_split.csv": "Random",
    "cross_subject_split.csv": "Cross-subject",
    "cross_device_dexcom_to_libre.csv": "Dexcom→Libre",
    "cross_device_libre_to_dexcom.csv": "Libre→Dexcom",
    "cross_setting_mealtype_holdout_breakfast.csv": "Breakfast holdout",
}


def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(x))


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
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "Pearson": pearson(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
    }


def get_model(name, seed):
    name = name.lower()
    if name in ["histgradientboosting", "hgb"]:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return "HistGradientBoosting", HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=seed
        )
    if name in ["randomforest", "rf"]:
        from sklearn.ensemble import RandomForestRegressor
        return "RandomForest", RandomForestRegressor(
            n_estimators=400, min_samples_leaf=3, random_state=seed, n_jobs=-1
        )
    if name in ["xgboost", "xgb"]:
        try:
            from xgboost import XGBRegressor
        except Exception as e:
            raise RuntimeError(f"XGBoost not installed: {e}")
        return "XGBoost", XGBRegressor(
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
    if name in ["ridge"]:
        from sklearn.linear_model import Ridge
        return "Ridge", Ridge(alpha=1.0)

    raise ValueError(f"Unknown model: {name}")


def try_get_transformed_feature_names(preprocess, X):
    try:
        return preprocess.get_feature_names_out(X.columns).tolist()
    except Exception:
        try:
            return preprocess.get_feature_names_out().tolist()
        except Exception:
            return [f"f{i}" for i in range(preprocess.transform(X.head(2)).shape[1])]


def run_shap_if_possible(pipe, X_test, model_name, split_name, output_dir, max_samples=500):
    """
    Optional SHAP analysis for tree-based transformed feature space.
    If shap is unavailable, silently skips.
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] SHAP unavailable, skip: {e}")
        return None

    try:
        preprocess = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]

        X_sample = X_test.sample(min(max_samples, len(X_test)), random_state=42)
        Xt = preprocess.transform(X_sample)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        feature_names = try_get_transformed_feature_names(preprocess, X_test)

        explainer = shap.Explainer(model, Xt, feature_names=feature_names)
        shap_values = explainer(Xt)

        shap_dir = output_dir / "shap"
        shap_dir.mkdir(parents=True, exist_ok=True)

        # Save mean absolute SHAP.
        vals = np.abs(shap_values.values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": vals})
        shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(shap_dir / f"shap_mean_abs_{safe_name(model_name)}_{safe_name(split_name)}.csv", index=False)

        # Beeswarm / summary plot.
        try:
            shap.plots.beeswarm(shap_values, max_display=25, show=False)
            plt.tight_layout()
            plt.savefig(shap_dir / f"shap_beeswarm_{safe_name(model_name)}_{safe_name(split_name)}.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"[WARN] Failed SHAP beeswarm: {e}")

        return shap_df
    except Exception as e:
        print(f"[WARN] SHAP failed for {model_name}/{split_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", default="HistGradientBoosting,XGBoost")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--output_dir", default="outputs/supplement_exp5")
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--max_test_samples", type=int, default=1000)
    parser.add_argument("--run_shap", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out = Path(config["output_dir"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta, seq, rel_grid = load_events(out)
    X, preprocess, numeric_cols, categorical_cols = build_tabular_matrix(meta, seq, rel_grid, config)
    target = config.get("main_target", "iauc_2h")
    y = pd.to_numeric(meta[target], errors="coerce")
    seed = int(config.get("random_seed", 42))

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    all_importances = []
    all_metrics = []

    for split_name in splits:
        split_path = out / "splits" / split_name
        if not split_path.exists():
            print(f"[WARN] Missing split: {split_path}")
            continue

        split = pd.read_csv(split_path)
        split_map = dict(zip(split["event_id"], split["split"]))
        split_label = meta["event_id"].map(split_map).fillna("unused")

        train_mask = (split_label == "train") & y.notna()
        test_mask = (split_label == "test") & y.notna()

        if train_mask.sum() < 20 or test_mask.sum() < 5:
            print(f"[WARN] Too few samples for {split_name}")
            continue

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]

        if len(X_test) > args.max_test_samples:
            idx = X_test.sample(args.max_test_samples, random_state=seed).index
            X_perm = X_test.loc[idx]
            y_perm = y_test.loc[idx]
        else:
            X_perm = X_test
            y_perm = y_test

        for model_key in models:
            try:
                model_name, model = get_model(model_key, seed)
            except Exception as e:
                print(f"[WARN] Skip model {model_key}: {e}")
                continue

            print(f"[RUN] {model_name} / {split_name}")
            pipe = Pipeline([
                ("preprocess", preprocess),
                ("model", model),
            ])
            pipe.fit(X_train, y_train)

            pred = pipe.predict(X_test)
            m = metrics(y_test, pred)
            m.update({
                "model": model_name,
                "split_file": split_name,
                "split_label": LABEL_MAP.get(split_name, split_name),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
            })
            all_metrics.append(m)
            print(f"[METRIC] RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}, R2={m['R2']:.3f}, r={m['Pearson']:.3f}, bias={m['bias']:.2f}")

            # Permutation importance on raw columns through the pipeline.
            try:
                perm = permutation_importance(
                    pipe,
                    X_perm,
                    y_perm,
                    n_repeats=args.n_repeats,
                    random_state=seed,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1,
                )
                imp = pd.DataFrame({
                    "feature": X_perm.columns,
                    "importance_mean_RMSE_increase": perm.importances_mean,
                    "importance_std": perm.importances_std,
                    "model": model_name,
                    "split_file": split_name,
                    "split_label": LABEL_MAP.get(split_name, split_name),
                    "n_perm_samples": len(X_perm),
                }).sort_values("importance_mean_RMSE_increase", ascending=False)

                imp_path = output_dir / f"permutation_importance_{safe_name(model_name)}_{safe_name(split_name)}.csv"
                imp.to_csv(imp_path, index=False)
                all_importances.append(imp)
            except Exception as e:
                print(f"[WARN] Permutation importance failed for {model_name}/{split_name}: {e}")

            if args.run_shap and model_name in ["XGBoost", "HistGradientBoosting", "RandomForest"]:
                run_shap_if_possible(pipe, X_test, model_name, split_name, output_dir)

    if all_importances:
        imp_all = pd.concat(all_importances, ignore_index=True)
        imp_all.to_csv(output_dir / "supplement_exp5_permutation_importance_all.csv", index=False)

        top = (
            imp_all.groupby(["model", "feature"])
            .agg(
                mean_importance=("importance_mean_RMSE_increase", "mean"),
                std_importance=("importance_mean_RMSE_increase", "std"),
                n_splits=("split_file", "nunique"),
            )
            .reset_index()
            .sort_values(["model", "mean_importance"], ascending=[True, False])
        )
        top.to_csv(output_dir / "supplement_exp5_feature_importance_overall.csv", index=False)

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(output_dir / "supplement_exp5_base_model_metrics.csv", index=False)

    print(f"[OK] Saved interpretability outputs to {output_dir}")


if __name__ == "__main__":
    main()
