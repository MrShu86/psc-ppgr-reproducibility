from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_meal_type_value(x) -> str:
    if pd.isna(x):
        return "unknown"
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    if s in {"breakfast", "break fast", "morning meal"} or "breakfast" in s:
        return "breakfast"
    if s in {"lunch", "midday meal"} or "lunch" in s:
        return "lunch"
    if s in {"dinner", "supper", "evening meal"} or "dinner" in s or "supper" in s:
        return "dinner"
    if s in {"snack", "snacks"} or "snack" in s:
        return "snack"
    if s in {"meal", "meals"}:
        return "meal"
    return s if s else "unknown"


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Pearson": np.nan}
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else np.nan
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2,
        "Pearson": pearson,
    }


def summarize_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_folds"] = len(sub)
        for m in ["MAE", "RMSE", "R2", "Pearson"]:
            if m in sub.columns:
                row[f"{m}_mean"] = sub[m].mean()
                row[f"{m}_std"] = sub[m].std()
                row[f"{m}_median"] = sub[m].median()
        rows.append(row)
    return pd.DataFrame(rows)
