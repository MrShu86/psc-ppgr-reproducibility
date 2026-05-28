from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_events(output_dir: str | Path):
    out = Path(output_dir)
    meta = pd.read_csv(out / "events_metadata.csv")
    arr = np.load(out / "events_sequences.npz")
    seq = arr["sequences"]
    rel_grid = arr["rel_grid"]
    return meta, seq, rel_grid


def _unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def make_handcrafted_sequence_features(seq: np.ndarray, rel_grid: np.ndarray, mode: str = "causal_premeal") -> pd.DataFrame:
    """
    Handcrafted CGM features for tabular baselines.

    mode='causal_premeal' is prediction-safe:
        uses only pre-meal CGM values, so it does not leak the future PPGR target.

    mode='oracle_full_window' is NOT prediction-safe:
        uses post-meal CGM features derived from the same window used to compute iAUC.
        Use it only as an upper-bound / sanity-check representation experiment.
    """
    out = {}
    pre = rel_grid <= 0
    post = rel_grid >= 0

    x_pre = seq[:, pre]
    out["pre_mean"] = np.nanmean(x_pre, axis=1)
    out["pre_std"] = np.nanstd(x_pre, axis=1)
    out["pre_min"] = np.nanmin(x_pre, axis=1)
    out["pre_max"] = np.nanmax(x_pre, axis=1)
    out["pre_last"] = x_pre[:, -1]
    out["pre_slope"] = (x_pre[:, -1] - x_pre[:, 0]) / max(1.0, float(rel_grid[pre][-1] - rel_grid[pre][0]))

    if mode == "oracle_full_window":
        x_post = seq[:, post]
        out["post_mean"] = np.nanmean(x_post, axis=1)
        out["post_std"] = np.nanstd(x_post, axis=1)
        out["post_min"] = np.nanmin(x_post, axis=1)
        out["post_max"] = np.nanmax(x_post, axis=1)

        baseline = np.nanmean(seq[:, pre], axis=1)
        delta = seq[:, post] - baseline[:, None]
        out["delta_max"] = np.nanmax(delta, axis=1)
        out["delta_mean"] = np.nanmean(delta, axis=1)
        out["delta_std"] = np.nanstd(delta, axis=1)

        post_t = rel_grid[post]
        early = (post_t >= 0) & (post_t <= 60)
        late = (post_t >= 60) & (post_t <= 180)
        out["early_delta_mean"] = np.nanmean(delta[:, early], axis=1)
        out["late_delta_mean"] = np.nanmean(delta[:, late], axis=1)

    return pd.DataFrame(out)


def _make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_tabular_matrix(meta: pd.DataFrame, seq: np.ndarray, rel_grid: np.ndarray, config: dict):
    feature_mode = config.get("feature_mode", "causal_premeal")
    if feature_mode not in {"causal_premeal", "oracle_full_window"}:
        raise ValueError("feature_mode must be 'causal_premeal' or 'oracle_full_window'.")

    seq_feat = make_handcrafted_sequence_features(seq, rel_grid, mode=feature_mode)
    df = pd.concat([meta.reset_index(drop=True), seq_feat.reset_index(drop=True)], axis=1)

    # Remove duplicated column names before sklearn ColumnTransformer.
    df = df.loc[:, ~df.columns.duplicated()].copy()

    numeric_candidates = []

    # Meal nutrition is available at prediction time.
    for c in config.get("nutrition_columns", []):
        if c in df.columns:
            numeric_candidates.append(c)

    # Prediction-safe CGM features from pre-meal window only by default.
    for c in seq_feat.columns:
        if c in df.columns:
            numeric_candidates.append(c)

    # Baseline glucose is computed from pre-meal values and is safe.
    for c in ["baseline_glucose", "missing_ratio"]:
        if c in df.columns:
            numeric_candidates.append(c)

    # Activity summaries:
    # pre30 is prediction-safe. post120 is future information and excluded by default.
    for c in df.columns:
        if c.endswith("_pre30_mean"):
            numeric_candidates.append(c)
        if feature_mode == "oracle_full_window" and c.endswith("_post120_mean"):
            numeric_candidates.append(c)

    # Subject-level variables, excluding subject_id.
    for c in df.columns:
        if c.startswith("subject_") and c != "subject_id":
            numeric_candidates.append(c)

    numeric_candidates = _unique_keep_order(numeric_candidates)

    numeric_cols = []
    categorical_cols = []

    for c in numeric_candidates:
        if c not in df.columns:
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().sum() > 0:
            df[c] = converted
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    explicit_categoricals = [
        "device",
        "meal_type",
        "setting_baseline_bin",
        "setting_activity_bin",
    ]

    for c in explicit_categoricals:
        if c in df.columns:
            categorical_cols.append(c)

    for c in df.columns:
        if c.startswith("subject_") and c not in numeric_cols and c != "subject_id":
            categorical_cols.append(c)

    numeric_cols = _unique_keep_order([c for c in numeric_cols if c in df.columns])
    categorical_cols = _unique_keep_order([c for c in categorical_cols if c in df.columns and c not in numeric_cols])

    forbidden = {
        "event_id", "paired_event_id", "subject_id", "meal_timestamp",
        "device_col", "pre_minutes", "post_minutes", "grid_minutes",

        # Target leakage / derived future response fields:
        "peak_rise", "iauc_2h", "iauc_3h", "time_to_peak",
        "recovery_slope", "hyper_duration_140", "post_mean",
    }
    if feature_mode == "oracle_full_window":
        # In oracle mode, keep post/delta handcrafted features but still remove explicit targets.
        forbidden = forbidden - {"post_mean"}

    numeric_cols = [c for c in numeric_cols if c not in forbidden]
    categorical_cols = [c for c in categorical_cols if c not in forbidden]

    X = df[numeric_cols + categorical_cols].copy()
    X = X.loc[:, ~X.columns.duplicated()].copy()

    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))

    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_onehot_encoder()),
            ]),
            categorical_cols,
        ))

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

    print(f"[FEATURE MODE] {feature_mode}")
    print(f"[FEATURES] numeric={len(numeric_cols)} categorical={len(categorical_cols)}")
    print(f"[FEATURES] numeric columns: {numeric_cols}")
    if categorical_cols:
        print(f"[FEATURES] categorical columns: {categorical_cols}")

    return X, preprocess, numeric_cols, categorical_cols
