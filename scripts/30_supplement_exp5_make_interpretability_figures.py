import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_bar(df, x, y, title, ylabel, path, rotation=35):
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    xx = np.arange(len(df))
    plt.bar(xx, df[y])
    plt.axhline(0, linestyle="--")
    plt.xticks(xx, df[x], rotation=rotation, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp5_dir", default="outputs/supplement_exp5")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    exp5_dir = Path(args.exp5_dir)
    fig_dir = exp5_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Feature importance figures.
    imp_path = exp5_dir / "supplement_exp5_feature_importance_overall.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        for model, sub in imp.groupby("model"):
            sub = sub.sort_values("mean_importance", ascending=False).head(args.top_k).copy()
            sub = sub.sort_values("mean_importance", ascending=True)
            plt.figure(figsize=(8, 6))
            plt.barh(sub["feature"], sub["mean_importance"])
            plt.xlabel("Mean permutation importance (RMSE increase)")
            plt.title(f"Top features: {model}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"fig_exp5_feature_importance_{model}.png", dpi=300)
            plt.close()

    # Per-split feature importance for XGBoost/HGB.
    imp_all_path = exp5_dir / "supplement_exp5_permutation_importance_all.csv"
    if imp_all_path.exists():
        imp_all = pd.read_csv(imp_all_path)
        for (model, split_label), sub in imp_all.groupby(["model", "split_label"]):
            sub = sub.sort_values("importance_mean_RMSE_increase", ascending=False).head(15)
            sub = sub.sort_values("importance_mean_RMSE_increase", ascending=True)
            safe_split = str(split_label).replace("→", "to").replace(" ", "_").lower()
            safe_model = str(model).replace(" ", "_")
            plt.figure(figsize=(8, 6))
            plt.barh(sub["feature"], sub["importance_mean_RMSE_increase"])
            plt.xlabel("Permutation importance (RMSE increase)")
            plt.title(f"{model}: {split_label}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"fig_exp5_feature_importance_{safe_model}_{safe_split}.png", dpi=300)
            plt.close()

    # Residual group summaries.
    group_path = exp5_dir / "supplement_exp5_residual_group_summary.csv"
    if group_path.exists():
        group = pd.read_csv(group_path)
        # Plot baseline glucose bin and meal type effects.
        for dim in ["baseline_glucose_bin", "meal_type", "setting_baseline_bin"]:
            sub = group[group["dimension"].eq(dim)].copy()
            if sub.empty:
                continue
            # For each split, plot calibrated RMSE by model and dimension value.
            for split_label, ss in sub.groupby("split_label"):
                # Use only common key models for readability.
                ss = ss[ss["model_label"].isin(["HistGradientBoosting", "XGBoost", "METER-v1-TCN"])]
                if ss.empty:
                    continue
                ss["plot_label"] = ss["model_label"].astype(str) + "\n" + ss["dimension_value"].astype(str)
                ss = ss.sort_values(["model_label", "dimension_value"])
                safe_split = str(split_label).replace("→", "to").replace(" ", "_").lower()
                safe_dim = dim.replace(" ", "_")
                save_bar(
                    ss,
                    "plot_label",
                    "calibrated_RMSE",
                    f"Calibrated RMSE by {dim}: {split_label}",
                    "Calibrated RMSE",
                    fig_dir / f"fig_exp5_group_rmse_{safe_dim}_{safe_split}.png"
                )

    # Correlation heat style table figure for top correlations.
    corr_path = exp5_dir / "supplement_exp5_residual_feature_correlations.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        corr = corr[corr["residual_metric"].isin(["raw_abs_error", "calibrated_abs_error", "abs_error_reduction"])].copy()
        if not corr.empty:
            corr["abs_spearman"] = corr["spearman_r"].abs()
            top = corr.sort_values("abs_spearman", ascending=False).head(30).copy()
            top["plot_label"] = (
                top["split_label"].astype(str) + " | " +
                top["model_label"].astype(str) + " | " +
                top["feature"].astype(str) + " → " +
                top["residual_metric"].astype(str)
            )
            top = top.sort_values("abs_spearman", ascending=True)
            plt.figure(figsize=(10, 8))
            plt.barh(top["plot_label"], top["spearman_r"])
            plt.axvline(0, linestyle="--")
            plt.xlabel("Spearman correlation")
            plt.title("Top residual-feature correlations")
            plt.tight_layout()
            plt.savefig(fig_dir / "fig_exp5_top_residual_feature_correlations.png", dpi=300)
            plt.close()

    print(f"[OK] Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()
