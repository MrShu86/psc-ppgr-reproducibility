import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re


TIMESTAMP_CANDIDATES = [
    "timestamp", "time", "datetime", "date_time", "date time", "device_time",
    "device time", "start_time", "start time", "created_at", "created at",
    "event_time", "event time", "date", "DateTime", "Time"
]
GLUCOSE_CANDIDATES = [
    "glucose", "cgm", "sensor_glucose", "sensor glucose", "sgv", "bg",
    "blood_glucose", "blood glucose", "glucose_value", "glucose value",
    "CGM", "BGM", "Sensor Glucose"
]
CARB_CANDIDATES = [
    "carbs", "carb", "carbohydrate", "carbohydrates", "cho", "meal_carbs",
    "meal carbs", "carb_input", "carb input", "grams_carbs", "grams carbs",
    "Food", "food", "Carbohydrates"
]
INSULIN_CANDIDATES = [
    "insulin", "bolus", "basal", "dose", "units", "insulin_units",
    "insulin units", "total_dose", "total dose", "correction", "meal_bolus"
]
MODE_CANDIDATES = ["mode", "device_mode", "device mode", "activity", "state", "sleep", "exercise"]


def norm(s):
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def find_col(cols, candidates):
    norm_cols = [(norm(c), c) for c in cols]

    # exact first
    for cand in candidates:
        ncand = norm(cand)
        for nc, orig in norm_cols:
            if ncand == nc:
                return orig

    # contains next
    for cand in candidates:
        ncand = norm(cand)
        for nc, orig in norm_cols:
            if ncand in nc:
                return orig

    return None


def load_table(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in [".tsv", ".txt"]:
        try:
            return pd.read_csv(path, sep="\t", low_memory=False)
        except Exception:
            return pd.read_csv(path, low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file: {path}")


def infer_subject_from_path(path):
    """
    AZT1D is organized as:
      CGM Records / Subject 1 / Subject 1.csv
      CGM Records / Subject 2 / Subject 2.csv
      ...
    Therefore, the safest subject_id is extracted from the parent folder or filename.
    """
    parts = list(Path(path).parts)
    text_candidates = list(reversed(parts[-3:])) + [Path(path).stem]

    for text in text_candidates:
        m = re.search(r"Subject\s*[_\- ]*\s*(\d+)", str(text), flags=re.IGNORECASE)
        if m:
            return f"Subject {int(m.group(1))}"

    for text in text_candidates:
        m = re.search(r"(?:subj|subject|patient|participant|person|pt)\s*[_\- ]*\s*(\d+)", str(text), flags=re.IGNORECASE)
        if m:
            return f"Subject {int(m.group(1))}"

    return Path(path).stem


def to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce")


def numeric_safe(s):
    return pd.to_numeric(s, errors="coerce")


def candidate_files(root):
    root = Path(root)
    files = []
    for ext in ["*.csv", "*.tsv", "*.txt", "*.parquet", "*.xlsx", "*.xls"]:
        files.extend(root.rglob(ext))

    # Prefer the core data files under Subject folders; ignore visual/stat image folders automatically.
    keep = []
    for p in sorted(files):
        lower = str(p).lower()
        if "visual" in lower and "statistics" in lower:
            continue
        keep.append(p)
    return keep


def extract_tables(root, carb_threshold=5.0):
    cgm_frames = []
    carb_frames = []
    insulin_frames = []
    mode_frames = []
    logs = []

    for p in candidate_files(root):
        try:
            df = load_table(p)
            if df.empty:
                logs.append((str(p), "empty"))
                continue

            cols = list(df.columns)
            time_col = find_col(cols, TIMESTAMP_CANDIDATES)
            glucose_col = find_col(cols, GLUCOSE_CANDIDATES)
            carb_col = find_col(cols, CARB_CANDIDATES)
            insulin_col = find_col(cols, INSULIN_CANDIDATES)
            mode_col = find_col(cols, MODE_CANDIDATES)

            if time_col is None:
                logs.append((str(p), "skipped_no_time_col"))
                continue

            subject_id = infer_subject_from_path(p)
            subject_series = pd.Series([subject_id] * len(df), index=df.index)
            time_series = to_datetime_safe(df[time_col])

            if glucose_col is not None:
                temp = pd.DataFrame({
                    "subject_id": subject_series,
                    "timestamp": time_series,
                    "glucose": numeric_safe(df[glucose_col]),
                    "source_file": str(p),
                })
                temp = temp.dropna(subset=["timestamp", "glucose"])
                temp = temp[(temp["glucose"] >= 30) & (temp["glucose"] <= 500)]
                if not temp.empty:
                    cgm_frames.append(temp)

            if carb_col is not None:
                temp = pd.DataFrame({
                    "subject_id": subject_series,
                    "meal_time": time_series,
                    "carbs": numeric_safe(df[carb_col]),
                    "source_file": str(p),
                })
                temp = temp.dropna(subset=["meal_time", "carbs"])
                temp = temp[temp["carbs"] >= carb_threshold]
                if not temp.empty:
                    carb_frames.append(temp)

            # Event/value fallback for carb entries.
            event_col = None
            for c in cols:
                if any(k in norm(c) for k in ["event", "type", "description", "name"]):
                    event_col = c
                    break
            value_col = None
            for c in cols:
                if norm(c) in ["value", "amount", "grams", "quantity"]:
                    value_col = c
                    break

            if carb_col is None and event_col is not None and value_col is not None:
                event_text = df[event_col].astype(str).str.lower()
                mask = event_text.str.contains("carb|meal|food|cho", regex=True, na=False)
                temp = pd.DataFrame({
                    "subject_id": subject_series[mask],
                    "meal_time": time_series[mask],
                    "carbs": numeric_safe(df.loc[mask, value_col]),
                    "source_file": str(p),
                })
                temp = temp.dropna(subset=["meal_time", "carbs"])
                temp = temp[temp["carbs"] >= carb_threshold]
                if not temp.empty:
                    carb_frames.append(temp)

            if insulin_col is not None:
                temp = pd.DataFrame({
                    "subject_id": subject_series,
                    "timestamp": time_series,
                    "insulin": numeric_safe(df[insulin_col]),
                    "source_file": str(p),
                })
                temp = temp.dropna(subset=["timestamp", "insulin"])
                temp = temp[(temp["insulin"] > 0) & (temp["insulin"] < 100)]
                if not temp.empty:
                    insulin_frames.append(temp)

            if mode_col is not None:
                temp = pd.DataFrame({
                    "subject_id": subject_series,
                    "timestamp": time_series,
                    "mode": df[mode_col].astype(str),
                    "source_file": str(p),
                })
                temp = temp.dropna(subset=["timestamp"])
                if not temp.empty:
                    mode_frames.append(temp)

            logs.append((str(p), f"ok_subject={subject_id}; glucose_col={glucose_col}; carb_col={carb_col}; insulin_col={insulin_col}; mode_col={mode_col}"))

        except Exception as e:
            logs.append((str(p), f"error:{e}"))

    cgm = pd.concat(cgm_frames, ignore_index=True) if cgm_frames else pd.DataFrame()
    carbs = pd.concat(carb_frames, ignore_index=True) if carb_frames else pd.DataFrame()
    insulin = pd.concat(insulin_frames, ignore_index=True) if insulin_frames else pd.DataFrame()
    mode = pd.concat(mode_frames, ignore_index=True) if mode_frames else pd.DataFrame()

    return cgm, carbs, insulin, mode, pd.DataFrame(logs, columns=["file", "status"])


def meal_type_from_hour(h):
    if 5 <= h < 10:
        return "breakfast"
    if 10 <= h < 15:
        return "lunch"
    if 15 <= h < 21:
        return "dinner"
    return "snack"


def trapz_auc(times_min, values):
    if len(times_min) < 2:
        return np.nan
    order = np.argsort(times_min)
    return float(np.trapz(np.asarray(values)[order], np.asarray(times_min)[order]))


def nearest_mode(mode_df, subject_id, t):
    if mode_df.empty:
        return "unknown"
    sub = mode_df[mode_df["subject_id"].astype(str).eq(str(subject_id))]
    if sub.empty:
        return "unknown"
    sub = sub[sub["timestamp"] <= t].sort_values("timestamp")
    if sub.empty:
        return "unknown"
    dt = (t - sub.iloc[-1]["timestamp"]).total_seconds() / 3600
    if dt > 6:
        return "unknown"
    return str(sub.iloc[-1]["mode"])


def build_events(cgm, carbs, insulin, mode, pre_min=30, post_min=120, min_pre_points=3, min_post_points=12, clip_positive=True):
    rows = []

    if cgm.empty or carbs.empty:
        raise RuntimeError("CGM or carbohydrate table is empty. Check extraction logs and column names.")

    cgm["subject_id"] = cgm["subject_id"].astype(str)
    carbs["subject_id"] = carbs["subject_id"].astype(str)

    if not insulin.empty:
        insulin["subject_id"] = insulin["subject_id"].astype(str)
    if not mode.empty:
        mode["subject_id"] = mode["subject_id"].astype(str)

    cgm = cgm.sort_values(["subject_id", "timestamp"])
    carbs = carbs.sort_values(["subject_id", "meal_time"])

    event_id = 0
    for subject_id, meal_sub in carbs.groupby("subject_id"):
        cgm_sub = cgm[cgm["subject_id"].eq(subject_id)].sort_values("timestamp")
        if cgm_sub.empty:
            continue

        for _, meal in meal_sub.iterrows():
            t = meal["meal_time"]
            pre = cgm_sub[(cgm_sub["timestamp"] >= t - pd.Timedelta(minutes=pre_min)) & (cgm_sub["timestamp"] <= t)]
            post = cgm_sub[(cgm_sub["timestamp"] >= t) & (cgm_sub["timestamp"] <= t + pd.Timedelta(minutes=post_min))]

            if len(pre) < min_pre_points or len(post) < min_post_points:
                continue

            baseline = float(pre["glucose"].iloc[-1])
            pre_times = (pre["timestamp"] - pre["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 60.0
            pre_values = pre["glucose"].to_numpy(dtype=float)

            if len(pre_times) >= 2 and np.nanstd(pre_times) > 0:
                pre_slope = float(np.polyfit(pre_times, pre_values, 1)[0])
            else:
                pre_slope = 0.0

            post_times = (post["timestamp"] - t).dt.total_seconds().to_numpy() / 60.0
            post_values = post["glucose"].to_numpy(dtype=float)
            delta = post_values - baseline
            delta_for_iauc = np.maximum(delta, 0) if clip_positive else delta

            iauc_2h = trapz_auc(post_times, delta_for_iauc)
            auc_delta_2h = trapz_auc(post_times, delta)

            bolus_around_meal = np.nan
            insulin_pre30 = np.nan
            if not insulin.empty:
                ins_sub = insulin[insulin["subject_id"].eq(subject_id)]
                around = ins_sub[(ins_sub["timestamp"] >= t - pd.Timedelta(minutes=15)) & (ins_sub["timestamp"] <= t + pd.Timedelta(minutes=15))]
                preins = ins_sub[(ins_sub["timestamp"] >= t - pd.Timedelta(minutes=pre_min)) & (ins_sub["timestamp"] <= t)]
                bolus_around_meal = float(around["insulin"].sum()) if not around.empty else 0.0
                insulin_pre30 = float(preins["insulin"].sum()) if not preins.empty else 0.0

            hour = int(t.hour)
            event_id += 1
            rows.append({
                "event_id": f"azt1d_evt_{event_id:06d}",
                "subject_id": subject_id,
                "meal_time": t,
                "carbs": float(meal["carbs"]),
                "meal_type": meal_type_from_hour(hour),
                "hour": hour,
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "device_mode": nearest_mode(mode, subject_id, t),
                "baseline_glucose": baseline,
                "pre_mean": float(pre["glucose"].mean()),
                "pre_std": float(pre["glucose"].std(ddof=1)) if len(pre) > 1 else 0.0,
                "pre_min": float(pre["glucose"].min()),
                "pre_max": float(pre["glucose"].max()),
                "pre_last": baseline,
                "pre_slope": pre_slope,
                "bolus_around_meal": bolus_around_meal,
                "insulin_pre30": insulin_pre30,
                "iauc_pos_2h": iauc_2h,
                "auc_delta_2h": auc_delta_2h,
                "post_mean_delta_2h": float(np.nanmean(delta)),
                "post_max_delta_2h": float(np.nanmax(delta)),
                "post_min_delta_2h": float(np.nanmin(delta)),
                "n_pre_points": len(pre),
                "n_post_points": len(post),
                "source_file": meal.get("source_file", ""),
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="AZT1D CGM Records root folder or dataset root folder")
    parser.add_argument("--output_dir", default="outputs_azt1d")
    parser.add_argument("--carb_threshold", type=float, default=5.0)
    parser.add_argument("--pre_min", type=int, default=30)
    parser.add_argument("--post_min", type=int, default=120)
    parser.add_argument("--min_pre_points", type=int, default=3)
    parser.add_argument("--min_post_points", type=int, default=12)
    parser.add_argument("--allow_negative_iauc", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Extracting AZT1D tables with subject_id forced from Subject folders...")
    cgm, carbs, insulin, mode, logs = extract_tables(args.root, carb_threshold=args.carb_threshold)

    cgm.to_csv(output_dir / "azt1d_extracted_cgm.csv", index=False)
    carbs.to_csv(output_dir / "azt1d_extracted_carbs.csv", index=False)
    insulin.to_csv(output_dir / "azt1d_extracted_insulin.csv", index=False)
    mode.to_csv(output_dir / "azt1d_extracted_mode.csv", index=False)
    logs.to_csv(output_dir / "azt1d_extraction_logs.csv", index=False)

    print(f"[INFO] extracted: cgm={len(cgm)}, carbs={len(carbs)}, insulin={len(insulin)}, mode={len(mode)}")
    print(f"[INFO] subjects in CGM={cgm['subject_id'].nunique() if not cgm.empty else 0}, subjects in carbs={carbs['subject_id'].nunique() if not carbs.empty else 0}")

    events = build_events(
        cgm, carbs, insulin, mode,
        pre_min=args.pre_min,
        post_min=args.post_min,
        min_pre_points=args.min_pre_points,
        min_post_points=args.min_post_points,
        clip_positive=not args.allow_negative_iauc,
    )

    events_path = output_dir / "azt1d_meal_events.csv"
    events.to_csv(events_path, index=False)

    summary = {
        "n_events": len(events),
        "n_subjects": events["subject_id"].nunique() if not events.empty else 0,
        "target": "iauc_pos_2h" if not args.allow_negative_iauc else "auc_delta_2h",
        "pre_min": args.pre_min,
        "post_min": args.post_min,
        "carb_threshold": args.carb_threshold,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "azt1d_event_build_summary.csv", index=False)

    if not events.empty:
        events["meal_type"].value_counts().rename_axis("meal_type").reset_index(name="count").to_csv(
            output_dir / "azt1d_meal_type_counts.csv", index=False
        )
        events.groupby("subject_id").size().reset_index(name="n_events").to_csv(
            output_dir / "azt1d_subject_event_counts.csv", index=False
        )
        print("[OK] saved events:", events_path)
        print(pd.DataFrame([summary]).to_string(index=False))
        print("\n[SUBJECT EVENT COUNTS]")
        print(events.groupby("subject_id").size().sort_index().to_string())
        print("\n[MEAL TYPE COUNTS]")
        print(events["meal_type"].value_counts().to_string())
    else:
        print("[WARN] No events built. Check extraction logs.")

if __name__ == "__main__":
    main()
