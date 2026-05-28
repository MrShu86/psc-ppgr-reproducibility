import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_bar_plot(df, value_col, label_col, title, ylabel, output_path):
    if df.empty:
        return
    plot_df = df.copy()
    plt.figure(figsize=(10, 5))
    x = np.arange(len(plot_df))
    plt.bar(x, plot_df[value_col])
    plt.axhline(0, linestyle="--")
    plt.xticks(x, plot_df[label_col], rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp3_dir", default="outputs/supplement_exp3")
    args = parser.parse_args()

    exp3_dir = Path(args.exp3_dir)
    fig_dir = exp3_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    support_path = exp3_dir / "supplement_exp3_support_vs_no_update_tests.csv"
    psc_path = exp3_dir / "supplement_exp3_meter_psc_vs_tabular_fixed5_tests.csv"

    if support_path.exists():
        support = pd.read_csv(support_path)
        # Keep main robustness, 5-shot residual, top readable methods.
        main = support[
            support["split_file"].isin([
                "cross_device_dexcom_to_libre.csv",
                "cross_device_libre_to_dexcom.csv",
                "cross_subject_split.csv",
                "cross_setting_mealtype_holdout_breakfast.csv",
            ]) &
            support["shot"].eq(5) &
            support["tested"].eq("support_residual_calibration")
        ].copy()

        # For each split, show the best RMSE reduction method.
        if not main.empty:
            best = main.loc[main.groupby("split_file")["relative_RMSE_reduction_%"].idxmax()].copy()
            best["plot_label"] = best["split_label"] + "\n" + best["method"].astype(str)
            make_bar_plot(
                best,
                "relative_RMSE_reduction_%",
                "plot_label",
                "Best 5-shot residual support calibration effect by split",
                "RMSE reduction vs no-update (%)",
                fig_dir / "fig_exp3_best_support_effect_by_split.png"
            )

    if psc_path.exists():
        psc = pd.read_csv(psc_path)
        best = psc[psc["comparison"].str.contains("BEST_TABULAR", na=False)].copy()
        if not best.empty:
            best["plot_label"] = best["split_label"] + "\nvs " + best["baseline"].astype(str)
            make_bar_plot(
                best,
                "relative_RMSE_reduction_by_PSC_%",
                "plot_label",
                "METER-PSC vs best tabular support baseline",
                "Positive means METER-PSC lower RMSE (%)",
                fig_dir / "fig_exp3_meter_psc_vs_best_tabular.png"
            )

    print(f"[OK] Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()
