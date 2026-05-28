from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .io_utils import (
    canonicalize_columns,
    ensure_dir,
    find_participant_csvs,
    load_bio_table,
    numeric_series,
    safe_to_datetime,
)


def _get_time_grid(pre_minutes: int, post_minutes: int, grid_minutes: int) -> np.ndarray:
    return np.arange(-pre_minutes, post_minutes + 1, grid_minutes, dtype=float)


def _interp_window(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    meal_time: pd.Timestamp,
    rel_grid: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Extracts and interpolates a meal-centered glucose sequence.
    Returns sequence and missing ratio before interpolation on the grid.
    """
    start = meal_time + pd.Timedelta(minutes=float(rel_grid.min()))
    end = meal_time + pd.Timedelta(minutes=float(rel_grid.max()))

    sub = df[(df[time_col] >= start) & (df[time_col] <= end)][[time_col, value_col]].dropna()
    if len(sub) < 2:
        return np.full(len(rel_grid), np.nan, dtype=float), 1.0

    rel_min = (sub[time_col] - meal_time).dt.total_seconds().to_numpy() / 60.0
    vals = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(rel_min) & np.isfinite(vals)
    rel_min = rel_min[ok]
    vals = vals[ok]
    if len(vals) < 2:
        return np.full(len(rel_grid), np.nan, dtype=float), 1.0

    # Remove duplicates by averaging.
    tmp = pd.DataFrame({"rel": rel_min, "val": vals}).groupby("rel", as_index=False)["val"].mean()
    rel_min = tmp["rel"].to_numpy()
    vals = tmp["val"].to_numpy()

    interp = np.interp(rel_grid, rel_min, vals, left=np.nan, right=np.nan)
    missing_ratio = float(np.mean(~np.isfinite(interp)))

    # Fill boundary NaN by nearest valid values after measuring missingness.
    s = pd.Series(interp).interpolate(limit_direction="both")
    return s.to_numpy(dtype=float), missing_ratio


def _window_mean_activity(
    df: pd.DataFrame,
    time_col: str,
    col: str,
    meal_time: pd.Timestamp,
    start_min: int,
    end_min: int,
) -> float:
    if col not in df.columns:
        return np.nan
    start = meal_time + pd.Timedelta(minutes=start_min)
    end = meal_time + pd.Timedelta(minutes=end_min)
    sub = df[(df[time_col] >= start) & (df[time_col] <= end)]
    return pd.to_numeric(sub[col], errors="coerce").mean()


def compute_targets(seq: np.ndarray, rel_grid: np.ndarray, min_post_points: int = 12) -> Dict[str, float]:
    """
    Computes meal-centered PPGR targets from one glucose sequence.

    baseline: mean glucose from [-30, 0] min
    peak_rise: max post-meal glucose minus baseline
    iauc_2h: incremental area under curve from 0 to 120 min
    iauc_3h: incremental area under curve from 0 to 180 min
    time_to_peak: first post-meal peak time
    recovery_slope: slope from peak to final post-meal point
    hyper_duration_140: minutes above 140 mg/dL post meal
    """
    seq = np.asarray(seq, dtype=float)
    out = {
        "baseline_glucose": np.nan,
        "peak_rise": np.nan,
        "iauc_2h": np.nan,
        "iauc_3h": np.nan,
        "time_to_peak": np.nan,
        "recovery_slope": np.nan,
        "hyper_duration_140": np.nan,
        "post_mean": np.nan,
    }

    pre_mask = (rel_grid >= -30) & (rel_grid <= 0)
    post_mask = (rel_grid >= 0) & (rel_grid <= rel_grid.max())
    if np.sum(np.isfinite(seq[post_mask])) < min_post_points:
        return out

    baseline = np.nanmean(seq[pre_mask])
    if not np.isfinite(baseline):
        return out

    post_t = rel_grid[post_mask]
    post_g = seq[post_mask]
    delta = post_g - baseline

    out["baseline_glucose"] = float(baseline)
    out["post_mean"] = float(np.nanmean(post_g))
    out["peak_rise"] = float(np.nanmax(delta))

    valid = np.isfinite(delta)
    if np.any(valid):
        peak_idx = np.nanargmax(delta)
        out["time_to_peak"] = float(post_t[peak_idx])

        final_g = post_g[np.where(np.isfinite(post_g))[0][-1]]
        peak_g = post_g[peak_idx]
        denom = max(float(post_t[-1] - post_t[peak_idx]), 1e-6)
        out["recovery_slope"] = float((final_g - peak_g) / denom)

    for name, max_t in [("iauc_2h", 120), ("iauc_3h", 180)]:
        mask = (post_t >= 0) & (post_t <= max_t) & np.isfinite(delta)
        if np.sum(mask) >= 2:
            positive_delta = np.maximum(delta[mask], 0)
            out[name] = float(np.trapz(positive_delta, post_t[mask]))

    hyper = (post_g > 140) & np.isfinite(post_g)
    if len(post_t) >= 2:
        step = float(np.median(np.diff(post_t)))
        out["hyper_duration_140"] = float(np.sum(hyper) * step)

    return out


def build_meal_events(config: dict) -> None:
    data_root = Path(config["data_root"])
    output_dir = ensure_dir(config["output_dir"])
    pre = int(config.get("pre_minutes", 30))
    post = int(config.get("post_minutes", 180))
    grid = int(config.get("grid_minutes", 5))
    max_missing = float(config.get("max_missing_ratio", 0.30))
    min_post = int(config.get("min_post_points_for_target", 12))
    devices = config.get("devices", {"libre": "Libre GL", "dexcom": "Dexcom GL"})
    nutrition_cols = config.get("nutrition_columns", [])
    activity_cols = config.get("activity_columns", [])
    subject_cols = config.get("subject_columns", [])

    rel_grid = _get_time_grid(pre, post, grid)
    csvs = find_participant_csvs(data_root)
    if not csvs:
        raise RuntimeError(f"No CGMacros participant CSV files found under {data_root}")

    bio = load_bio_table(data_root)
    bio_lookup = {}
    if not bio.empty:
        for _, row in bio.iterrows():
            bio_lookup[row["subject_id"]] = row.to_dict()

    metadata_rows: List[dict] = []
    seqs: List[np.ndarray] = []

    for subject_id, csv_path in tqdm(csvs, desc="Building meal events"):
        df = pd.read_csv(csv_path)
        df = canonicalize_columns(df)

        if "Timestamp" not in df.columns:
            raise ValueError(f"Timestamp column not found in {csv_path}")

        df["Timestamp"] = safe_to_datetime(df["Timestamp"])
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

        if "Meal Type" not in df.columns:
            continue

        meal_mask = df["Meal Type"].notna() & (df["Meal Type"].astype(str).str.strip() != "")
        meal_df = df[meal_mask].copy()
        if meal_df.empty:
            continue

        subj_info = bio_lookup.get(subject_id, {})

        for meal_idx, meal_row in meal_df.iterrows():
            meal_time = meal_row["Timestamp"]
            meal_type = str(meal_row.get("Meal Type", "")).strip()
            paired_event_id = f"{subject_id}_meal_{len(metadata_rows):06d}"

            # Nutrition values at meal timestamp.
            meal_features = {}
            for c in nutrition_cols:
                meal_features[c] = pd.to_numeric(pd.Series([meal_row.get(c, np.nan)]), errors="coerce").iloc[0]

            # Activity context around meal.
            activity_features = {}
            for c in activity_cols:
                activity_features[f"{c}_pre30_mean"] = _window_mean_activity(df, "Timestamp", c, meal_time, -30, 0)
                activity_features[f"{c}_post120_mean"] = _window_mean_activity(df, "Timestamp", c, meal_time, 0, 120)

            for device_name, device_col in devices.items():
                if device_col not in df.columns:
                    continue

                seq, missing_ratio = _interp_window(df, "Timestamp", device_col, meal_time, rel_grid)
                if missing_ratio > max_missing:
                    continue

                targets = compute_targets(seq, rel_grid, min_post_points=min_post)
                if not np.isfinite(targets.get("iauc_2h", np.nan)):
                    continue

                row = {
                    "event_id": f"{subject_id}_{device_name}_{len(seqs):06d}",
                    "paired_event_id": paired_event_id,
                    "subject_id": subject_id,
                    "device": device_name,
                    "device_col": device_col,
                    "meal_timestamp": meal_time,
                    "meal_type": meal_type,
                    "missing_ratio": missing_ratio,
                    "pre_minutes": pre,
                    "post_minutes": post,
                    "grid_minutes": grid,
                }
                row.update(meal_features)
                row.update(activity_features)
                row.update(targets)

                # Add selected subject metadata.
                for c in subject_cols:
                    row[f"subject_{c}"] = subj_info.get(c, np.nan)

                # Derived setting labels.
                row["setting_meal_type"] = meal_type.lower()
                row["setting_baseline_bin"] = np.nan
                row["setting_activity_bin"] = np.nan

                metadata_rows.append(row)
                seqs.append(seq.astype(np.float32))

    meta = pd.DataFrame(metadata_rows)
    if meta.empty:
        raise RuntimeError("No valid meal events were built. Check path, columns, and missing thresholds.")

    # Derived tertile-based settings.
    if "baseline_glucose" in meta.columns:
        try:
            meta["setting_baseline_bin"] = pd.qcut(meta["baseline_glucose"], 3, labels=["low", "mid", "high"], duplicates="drop")
        except Exception:
            pass

    # Use Mets_post120_mean first; fallback to activity calories.
    act_col = None
    for c in ["Mets_post120_mean", "Calories (Activity)_post120_mean", "HR_post120_mean"]:
        if c in meta.columns and meta[c].notna().sum() > 0:
            act_col = c
            break
    if act_col:
        try:
            meta["setting_activity_bin"] = pd.qcut(meta[act_col], 3, labels=["low", "mid", "high"], duplicates="drop")
        except Exception:
            pass

    seq_arr = np.stack(seqs, axis=0)
    meta.to_csv(output_dir / "events_metadata.csv", index=False)
    np.savez_compressed(output_dir / "events_sequences.npz", sequences=seq_arr, rel_grid=rel_grid)

    print(f"[OK] Built {len(meta)} device-level meal events")
    print(f"[OK] Unique paired meal events: {meta['paired_event_id'].nunique()}")
    print(f"[OK] Subjects: {meta['subject_id'].nunique()}")
    print(f"[OK] Devices: {meta['device'].value_counts().to_dict()}")
    print(f"[OK] Saved metadata: {output_dir / 'events_metadata.csv'}")
    print(f"[OK] Saved sequences: {output_dir / 'events_sequences.npz'}")
