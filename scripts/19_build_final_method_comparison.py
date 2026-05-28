import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", default="outputs/summary")
    parser.add_argument("--output", default="outputs/summary/final_method_comparison_table.csv")
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    seq_path = summary_dir / "sequence_baseline_summary.csv"
    if seq_path.exists():
        seq = pd.read_csv(seq_path)
        for _, r in seq.iterrows():
            rows.append({
                "method": f"Sequence-{r.get('model', 'unknown')}",
                "split_file": r.get("split_file"),
                "setting": "0-shot",
                "shot": 0,
                "RMSE": r.get("RMSE"),
                "MAE": r.get("MAE"),
                "R2": r.get("R2"),
                "Pearson": r.get("Pearson"),
                "source": seq_path.name,
            })

    tab_path = summary_dir / "best_tabular_result_per_split.csv"
    if tab_path.exists():
        tab = pd.read_csv(tab_path)
        for _, r in tab.iterrows():
            rows.append({
                "method": f"Tabular-{r.get('model', 'unknown')}",
                "split_file": r.get("split_file"),
                "setting": "0-shot",
                "shot": 0,
                "RMSE": r.get("RMSE"),
                "MAE": r.get("MAE"),
                "R2": r.get("R2"),
                "Pearson": r.get("Pearson"),
                "source": tab_path.name,
            })

    support_path = summary_dir / "support_calibration_best_overall_by_split.csv"
    if support_path.exists():
        sup = pd.read_csv(support_path)
        for _, r in sup.iterrows():
            rows.append({
                "method": f"{r.get('method')}-{r.get('encoder')}",
                "split_file": r.get("split_file"),
                "setting": r.get("personalization"),
                "shot": r.get("shot"),
                "RMSE": r.get("RMSE_mean"),
                "MAE": r.get("MAE_mean"),
                "R2": r.get("R2_mean"),
                "Pearson": r.get("Pearson_mean"),
                "source": support_path.name,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No comparison inputs found.")

    label_map = {
        "random_meal_split.csv": "Random",
        "cross_subject_split.csv": "Cross-subject",
        "cross_device_dexcom_to_libre.csv": "Dexcom→Libre",
        "cross_device_libre_to_dexcom.csv": "Libre→Dexcom",
        "cross_setting_mealtype_holdout_breakfast.csv": "Breakfast holdout",
    }
    df["split_label"] = df["split_file"].map(label_map).fillna(df["split_file"])
    df = df.sort_values(["split_label", "RMSE"])
    df.to_csv(output, index=False)

    best = df.loc[df.groupby("split_file")["RMSE"].idxmin()].copy()
    best_path = output.parent / "final_best_method_by_split.csv"
    best.to_csv(best_path, index=False)

    print(f"[OK] Saved final comparison: {output}")
    print(f"[OK] Saved final best by split: {best_path}")
    print("\n[FINAL BEST BY SPLIT]")
    print(best[["split_label", "method", "setting", "shot", "RMSE", "R2", "Pearson"]].to_string(index=False))


if __name__ == "__main__":
    main()
