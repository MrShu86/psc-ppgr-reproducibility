import argparse
from pathlib import Path
import pandas as pd
import numpy as np


LABEL_MAP = {
    "random_meal_split.csv": "Random",
    "cross_subject_split.csv": "Cross-subject",
    "cross_device_dexcom_to_libre.csv": "Dexcom→Libre",
    "cross_device_libre_to_dexcom.csv": "Libre→Dexcom",
    "cross_setting_mealtype_holdout_breakfast.csv": "Breakfast holdout",
}

ROBUSTNESS_SPLITS = [
    "cross_device_dexcom_to_libre.csv",
    "cross_device_libre_to_dexcom.csv",
    "cross_subject_split.csv",
    "cross_setting_mealtype_holdout_breakfast.csv",
]

MAIN_SPLITS = [
    "random_meal_split.csv",
    "cross_device_dexcom_to_libre.csv",
    "cross_device_libre_to_dexcom.csv",
    "cross_subject_split.csv",
    "cross_setting_mealtype_holdout_breakfast.csv",
]


def _read_csv_if_exists(path: Path):
    return pd.read_csv(path) if path.exists() else None


def _metric_row(method, split_file, setting, shot, row, source):
    return {
        "method": method,
        "split_file": split_file,
        "split_label": LABEL_MAP.get(split_file, split_file),
        "setting": setting,
        "shot": shot,
        "RMSE": row.get("RMSE", np.nan),
        "MAE": row.get("MAE", np.nan),
        "R2": row.get("R2", np.nan),
        "Pearson": row.get("Pearson", np.nan),
        "source": source,
    }


def collect_tabular(summary_dir: Path):
    rows = []
    p = summary_dir / "best_tabular_result_per_split.csv"
    df = _read_csv_if_exists(p)
    if df is None:
        return rows
    for _, r in df.iterrows():
        split_file = r.get("split_file")
        if split_file not in MAIN_SPLITS:
            continue
        method = f"Best tabular ({r.get('model', 'unknown')})"
        rows.append(_metric_row(method, split_file, "0-shot", 0, r, p.name))
    return rows


def collect_sequence(summary_dir: Path, results_dir: Path):
    rows = []
    p = summary_dir / "sequence_baseline_summary.csv"
    df = _read_csv_if_exists(p)
    if df is not None:
        for _, r in df.iterrows():
            split_file = r.get("split_file")
            if split_file not in MAIN_SPLITS:
                continue
            method = f"{str(r.get('model', 'Sequence')).upper()} sequence"
            rows.append(_metric_row(method, split_file, "0-shot", 0, r, p.name))
        return rows

    raw_files = sorted(results_dir.glob("sequence_baseline_*_iauc_2h.csv"))
    raw_files = [x for x in raw_files if "predictions" not in x.name]
    if not raw_files:
        return rows
    raw = pd.concat([pd.read_csv(x).assign(source_file=x.name) for x in raw_files], ignore_index=True)
    raw = raw[raw["split_file"].isin(MAIN_SPLITS)]
    for _, r in raw.iterrows():
        method = f"{str(r.get('model', 'Sequence')).upper()} sequence"
        rows.append(_metric_row(method, r.get("split_file"), "0-shot", 0, r, r.get("source_file")))
    return rows


def collect_meter_zero_shot(summary_dir: Path, results_dir: Path, method_name="METER-v1", encoder="tcn"):
    rows = []
    p = summary_dir / "support_calibration_all_summary.csv"
    df = _read_csv_if_exists(p)
    if df is not None:
        sub = df[
            (df["method"].astype(str) == method_name) &
            (df["encoder"].astype(str) == encoder) &
            (df["personalization"].astype(str) == "global_0shot") &
            (df["shot"].astype(int) == 0) &
            (df["split_file"].isin(MAIN_SPLITS))
        ].copy()
        for _, r in sub.iterrows():
            split_file = r["split_file"]
            rows.append({
                "method": f"{method_name}-{encoder.upper()} 0-shot",
                "split_file": split_file,
                "split_label": LABEL_MAP.get(split_file, split_file),
                "setting": "0-shot",
                "shot": 0,
                "RMSE": r.get("RMSE_mean", np.nan),
                "MAE": r.get("MAE_mean", np.nan),
                "R2": r.get("R2_mean", np.nan),
                "Pearson": r.get("Pearson_mean", np.nan),
                "source": p.name,
                "n_subjects": r.get("n_subjects", np.nan),
                "n_query_total": r.get("n_query_total", np.nan),
            })
        return rows

    raw_files = sorted(results_dir.glob(f"meter_v1_{encoder}_*_iauc_2h.csv"))
    raw_files = [x for x in raw_files if not any(s in x.name for s in ["predictions", "repr", "support"])]
    for x in raw_files:
        df0 = pd.read_csv(x)
        for _, r in df0.iterrows():
            split_file = r.get("split_file")
            if split_file in MAIN_SPLITS:
                rows.append(_metric_row(f"{method_name}-{encoder.upper()} 0-shot", split_file, "0-shot", 0, r, x.name))
    return rows


def collect_fixed_psc(summary_dir: Path, method_name="METER-v1", encoder="tcn", shot=5, calibration="support_residual_calibration"):
    rows = []
    p = summary_dir / "support_calibration_all_summary.csv"
    df = _read_csv_if_exists(p)
    if df is None:
        return rows

    sub = df[
        (df["method"].astype(str) == method_name) &
        (df["encoder"].astype(str) == encoder) &
        (df["personalization"].astype(str) == calibration) &
        (df["shot"].astype(int) == shot) &
        (df["split_file"].isin(MAIN_SPLITS))
    ].copy()

    for _, r in sub.iterrows():
        split_file = r["split_file"]
        rows.append({
            "method": f"METER-PSC ({method_name}-{encoder.upper()} + {shot}-shot residual)",
            "split_file": split_file,
            "split_label": LABEL_MAP.get(split_file, split_file),
            "setting": calibration,
            "shot": shot,
            "RMSE": r.get("RMSE_mean", np.nan),
            "MAE": r.get("MAE_mean", np.nan),
            "R2": r.get("R2_mean", np.nan),
            "Pearson": r.get("Pearson_mean", np.nan),
            "source": p.name,
            "n_subjects": r.get("n_subjects", np.nan),
            "n_query_total": r.get("n_query_total", np.nan),
        })
    return rows


def collect_best_support(summary_dir: Path):
    rows = []
    p = summary_dir / "support_calibration_best_overall_by_split.csv"
    df = _read_csv_if_exists(p)
    if df is None:
        return rows
    df = df[df["split_file"].isin(MAIN_SPLITS)].copy()
    for _, r in df.iterrows():
        split_file = r["split_file"]
        rows.append({
            "method": f"Best support ({r.get('method')}-{str(r.get('encoder')).upper()})",
            "split_file": split_file,
            "split_label": LABEL_MAP.get(split_file, split_file),
            "setting": r.get("personalization"),
            "shot": r.get("shot"),
            "RMSE": r.get("RMSE_mean", np.nan),
            "MAE": r.get("MAE_mean", np.nan),
            "R2": r.get("R2_mean", np.nan),
            "Pearson": r.get("Pearson_mean", np.nan),
            "source": p.name,
            "n_subjects": r.get("n_subjects", np.nan),
            "n_query_total": r.get("n_query_total", np.nan),
        })
    return rows


def add_improvement_columns(df):
    df = df.copy()
    baseline_rows = df[df["method"].str.contains("METER-v1", na=False) & df["method"].str.contains("0-shot", na=False)]
    baseline_map = dict(zip(baseline_rows["split_file"], baseline_rows["RMSE"]))

    seq = df[df["method"].str.contains("sequence", case=False, na=False)]
    if not seq.empty:
        seq_best = seq.loc[seq.groupby("split_file")["RMSE"].idxmin()]
        seq_map = dict(zip(seq_best["split_file"], seq_best["RMSE"]))
    else:
        seq_map = {}

    imp_meter = []
    imp_seq = []
    for _, r in df.iterrows():
        split = r["split_file"]
        rmse = r["RMSE"]
        base = baseline_map.get(split, np.nan)
        seq_base = seq_map.get(split, np.nan)
        imp_meter.append(np.nan if not np.isfinite(base) else (base - rmse) / base * 100)
        imp_seq.append(np.nan if not np.isfinite(seq_base) else (seq_base - rmse) / seq_base * 100)
    df["RMSE_improvement_vs_METERv1_0shot_%"] = imp_meter
    df["RMSE_improvement_vs_best_sequence_%"] = imp_seq
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", default="outputs/summary")
    parser.add_argument("--results_dir", default="outputs/results")
    parser.add_argument("--output_dir", default="outputs/supplement_exp1")
    parser.add_argument("--psc_method", default="METER-v1")
    parser.add_argument("--psc_encoder", default="tcn")
    parser.add_argument("--psc_shot", type=int, default=5)
    parser.add_argument("--psc_calibration", default="support_residual_calibration")
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows += collect_tabular(summary_dir)
    rows += collect_sequence(summary_dir, results_dir)
    rows += collect_meter_zero_shot(summary_dir, results_dir, method_name=args.psc_method, encoder=args.psc_encoder)
    rows += collect_fixed_psc(summary_dir, method_name=args.psc_method, encoder=args.psc_encoder, shot=args.psc_shot, calibration=args.psc_calibration)
    rows += collect_best_support(summary_dir)

    if not rows:
        raise RuntimeError("No rows collected. Please check outputs/summary and outputs/results.")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["method", "split_file", "setting", "shot"], keep="first")
    df = add_improvement_columns(df)

    long_path = output_dir / "supplement_exp1_unified_long_results.csv"
    df.sort_values(["split_file", "RMSE"]).to_csv(long_path, index=False)

    main_df = df[
        df["method"].str.startswith("Best tabular") |
        df["method"].str.startswith("GRU sequence") |
        df["method"].str.startswith("TCN sequence") |
        df["method"].eq(f"{args.psc_method}-{args.psc_encoder.upper()} 0-shot") |
        df["method"].str.startswith("METER-PSC")
    ].copy()

    main_df["split_order"] = main_df["split_file"].map({s: i for i, s in enumerate(MAIN_SPLITS)})
    main_df = main_df.sort_values(["split_order", "method"])
    main_path = output_dir / "supplement_exp1_main_table.csv"
    main_df.to_csv(main_path, index=False)

    psc = df[df["method"].str.startswith("METER-PSC") & df["split_file"].isin(ROBUSTNESS_SPLITS)].copy()
    if not psc.empty:
        avg = {
            "method": psc["method"].iloc[0],
            "splits": "Dexcom→Libre; Libre→Dexcom; Cross-subject; Breakfast",
            "mean_RMSE": psc["RMSE"].mean(),
            "mean_MAE": psc["MAE"].mean(),
            "mean_R2": psc["R2"].mean(),
            "mean_Pearson": psc["Pearson"].mean(),
            "mean_improvement_vs_METERv1_0shot_%": psc["RMSE_improvement_vs_METERv1_0shot_%"].mean(),
            "mean_improvement_vs_best_sequence_%": psc["RMSE_improvement_vs_best_sequence_%"].mean(),
        }
        pd.DataFrame([avg]).to_csv(output_dir / "supplement_exp1_psc_robustness_average.csv", index=False)

    pivot = main_df.pivot_table(index="method", columns="split_label", values="RMSE", aggfunc="min")
    col_order = [LABEL_MAP[s] for s in MAIN_SPLITS if LABEL_MAP[s] in pivot.columns]
    pivot = pivot[col_order]
    pivot.to_csv(output_dir / "supplement_exp1_rmse_pivot_table.csv")

    print(f"[OK] Saved long results: {long_path}")
    print(f"[OK] Saved main table: {main_path}")
    print(f"[OK] Saved RMSE pivot: {output_dir / 'supplement_exp1_rmse_pivot_table.csv'}")
    print("\n[MAIN TABLE PREVIEW]")
    cols = ["split_label", "method", "setting", "shot", "RMSE", "MAE", "R2", "Pearson",
            "RMSE_improvement_vs_METERv1_0shot_%", "RMSE_improvement_vs_best_sequence_%"]
    print(main_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
