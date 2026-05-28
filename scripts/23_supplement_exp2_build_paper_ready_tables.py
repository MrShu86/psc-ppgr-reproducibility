import argparse
from pathlib import Path
import pandas as pd
import numpy as np


ROBUSTNESS_SPLITS = [
    "cross_device_dexcom_to_libre.csv",
    "cross_device_libre_to_dexcom.csv",
    "cross_subject_split.csv",
    "cross_setting_mealtype_holdout_breakfast.csv",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp2_dir", default="outputs/supplement_exp2")
    parser.add_argument("--output", default="outputs/supplement_exp2/supplement_exp2_paper_ready_tables.xlsx")
    args = parser.parse_args()

    exp2_dir = Path(args.exp2_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    summary_path = exp2_dir / "supplement_exp2_support_all_summary.csv"
    comp_path = exp2_dir / "supplement_exp2_meter_psc_vs_best_tabular_support.csv"
    best_path = exp2_dir / "supplement_exp2_best_overall_by_split.csv"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = pd.read_csv(summary_path)
    comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
    best = pd.read_csv(best_path) if best_path.exists() else pd.DataFrame()

    # Fixed 5-shot residual table for major methods.
    fixed = summary[
        (summary["shot"].eq(5)) &
        (summary["personalization"].eq("support_residual_calibration"))
    ].copy()

    # Keep representative methods.
    fixed["method_full"] = fixed["method"].astype(str) + "-" + fixed["encoder"].astype(str)
    keep_patterns = ["Tabular-XGBoost", "Tabular-HistGradientBoosting", "Tabular-RandomForest",
                     "Sequence-gru", "Sequence-tcn", "METER-v1-tcn", "METER-v2-tcn"]
    fixed_keep = fixed[fixed["method_full"].isin(keep_patterns) | fixed["method"].str.startswith("Tabular-", na=False)].copy()

    pivot_rmse = fixed_keep.pivot_table(index="method_full", columns="split_label", values="RMSE_mean", aggfunc="min")

    robustness = fixed[fixed["split_file"].isin(ROBUSTNESS_SPLITS)].copy()
    avg = (
        robustness.groupby(["method", "encoder", "personalization", "shot"])
        .agg(
            mean_RMSE=("RMSE_mean", "mean"),
            mean_MAE=("MAE_mean", "mean"),
            mean_R2=("R2_mean", "mean"),
            mean_Pearson=("Pearson_mean", "mean"),
            n_settings=("split_file", "nunique"),
        )
        .reset_index()
        .sort_values("mean_RMSE")
    )

    with pd.ExcelWriter(output) as writer:
        summary.to_excel(writer, sheet_name="all_support_summary", index=False)
        if not comp.empty:
            comp.to_excel(writer, sheet_name="METER_PSC_vs_tabular", index=False)
        if not best.empty:
            best.to_excel(writer, sheet_name="best_overall", index=False)
        fixed.to_excel(writer, sheet_name="fixed_5shot_residual", index=False)
        pivot_rmse.to_excel(writer, sheet_name="pivot_RMSE_5shot")
        avg.to_excel(writer, sheet_name="robustness_average", index=False)

    print(f"[OK] Saved paper-ready Excel: {output}")
    if not comp.empty:
        print("\n[METER-PSC vs Best Tabular Support]")
        print(comp.to_string(index=False))
    print("\n[Robustness average, top methods]")
    print(avg.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
