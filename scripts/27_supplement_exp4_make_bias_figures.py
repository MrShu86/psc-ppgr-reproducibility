import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROBUSTNESS_SPLITS = [
    "cross_device_dexcom_to_libre.csv",
    "cross_device_libre_to_dexcom.csv",
    "cross_subject_split.csv",
    "cross_setting_mealtype_holdout_breakfast.csv",
]


def save_bar(df, x_col, y_col, title, ylabel, output_path, rotation=35):
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    x = np.arange(len(df))
    plt.bar(x, df[y_col])
    plt.axhline(0, linestyle="--")
    plt.xticks(x, df[x_col], rotation=rotation, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_line(df, x_col, y_col, group_col, title, ylabel, output_path):
    if df.empty:
        return
    plt.figure(figsize=(8, 5))
    for group, sub in df.groupby(group_col):
        sub = sub.sort_values(x_col)
        plt.plot(sub[x_col], sub[y_col], marker="o", label=str(group))
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_hist(before, after, title, output_path):
    if len(before) == 0 or len(after) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(before, bins=30, alpha=0.5, label="Before calibration")
    plt.hist(after, bins=30, alpha=0.5, label="After calibration")
    plt.axvline(0, linestyle="--")
    plt.xlabel("Prediction error")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_scatter(df, title, output_path):
    if df.empty:
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(df["y_true"], df["y_pred"], s=12, alpha=0.35, label="Before")
    plt.scatter(df["y_true"], df["y_pred_calibrated"], s=12, alpha=0.35, label="After")
    mn = min(df["y_true"].min(), df["y_pred"].min(), df["y_pred_calibrated"].min())
    mx = max(df["y_true"].max(), df["y_pred"].max(), df["y_pred_calibrated"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True iAUC")
    plt.ylabel("Predicted iAUC")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp4_dir", default="outputs/supplement_exp4")
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--calibration", default="support_residual_calibration")
    parser.add_argument("--models", default="HistGradientBoosting,XGBoost,METER-v1-TCN")
    args = parser.parse_args()

    exp4_dir = Path(args.exp4_dir)
    fig_dir = exp4_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary_path = exp4_dir / "supplement_exp4_bias_summary_by_method_split.csv"
    detail_path = exp4_dir / "supplement_exp4_calibrated_query_predictions.csv"
    dim_path = exp4_dir / "supplement_exp4_bias_summary_by_dimension.csv"
    subj_path = exp4_dir / "supplement_exp4_subject_level_bias_before_after.csv"

    if not summary_path.exists() or not detail_path.exists():
        raise FileNotFoundError("Run script 26 first.")

    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    summary_main = summary[
        summary["shot"].eq(args.shot) &
        summary["calibration"].eq(args.calibration) &
        summary["split_file"].isin(ROBUSTNESS_SPLITS)
    ].copy()

    if models != ["all"]:
        summary_main = summary_main[summary_main["model_label"].isin(models)]

    # Figure 1: bias before/after by split/model.
    bias_rows = []
    for _, r in summary_main.iterrows():
        bias_rows.append({
            "plot_label": f"{r['split_label']}\n{r['model_label']}",
            "raw_bias": r["raw_bias"],
            "calibrated_bias": r["calibrated_bias"],
            "bias_reduction_abs": r["bias_reduction_abs"],
        })
    bias_df = pd.DataFrame(bias_rows)
    if not bias_df.empty:
        # before/after grouped bars
        plt.figure(figsize=(12, 5))
        x = np.arange(len(bias_df))
        w = 0.38
        plt.bar(x - w/2, bias_df["raw_bias"], w, label="Before")
        plt.bar(x + w/2, bias_df["calibrated_bias"], w, label="After")
        plt.axhline(0, linestyle="--")
        plt.xticks(x, bias_df["plot_label"], rotation=35, ha="right")
        plt.ylabel("Bias: prediction - true")
        plt.title(f"Bias before vs after {args.shot}-shot residual calibration")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_exp4_bias_before_after_by_split_model.png", dpi=300)
        plt.close()

        save_bar(
            bias_df,
            "plot_label",
            "bias_reduction_abs",
            f"Absolute bias reduction after {args.shot}-shot residual calibration",
            "Absolute bias reduction",
            fig_dir / "fig_exp4_abs_bias_reduction.png"
        )

    # Figure 2: MAE reduction by split/model.
    if not summary_main.empty:
        plot = summary_main.copy()
        plot["plot_label"] = plot["split_label"] + "\n" + plot["model_label"]
        save_bar(
            plot,
            "plot_label",
            "MAE_reduction_%",
            f"MAE reduction after {args.shot}-shot residual calibration",
            "MAE reduction (%)",
            fig_dir / "fig_exp4_mae_reduction_percent.png"
        )

    # Figure 3: shot curves if available.
    shot_curve = summary[
        summary["calibration"].eq(args.calibration) &
        summary["split_file"].isin(ROBUSTNESS_SPLITS)
    ].copy()
    if models != ["all"]:
        shot_curve = shot_curve[shot_curve["model_label"].isin(models)]

    # Save one shot curve per split.
    for split_label, sub in shot_curve.groupby("split_label"):
        if sub.empty:
            continue
        plt.figure(figsize=(8, 5))
        for model, ss in sub.groupby("model_label"):
            ss = ss.sort_values("shot")
            plt.plot(ss["shot"], ss["RMSE"], marker="o", label=model)
        plt.xlabel("Number of support meal events")
        plt.ylabel("RMSE")
        plt.title(f"Support-shot RMSE curve: {split_label}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        safe = str(split_label).replace("→", "to").replace(" ", "_").lower()
        plt.savefig(fig_dir / f"fig_exp4_shot_curve_{safe}.png", dpi=300)
        plt.close()

    # Figure 4: residual histograms and scatter plots for key split/model pairs.
    detail_main = detail[
        detail["shot"].eq(args.shot) &
        detail["calibration"].eq(args.calibration) &
        detail["split_file"].isin(ROBUSTNESS_SPLITS)
    ].copy()
    if models != ["all"]:
        detail_main = detail_main[detail_main["model_label"].isin(models)]

    for (split_label, model_label), sub in detail_main.groupby(["split_label", "model_label"]):
        if len(sub) < 20:
            continue
        safe_split = str(split_label).replace("→", "to").replace(" ", "_").lower()
        safe_model = str(model_label).replace(" ", "_").replace("/", "_").replace("→", "to")
        save_hist(
            sub["error_before"].dropna().to_numpy(),
            sub["error_after"].dropna().to_numpy(),
            f"Residual distribution: {split_label} / {model_label}",
            fig_dir / f"fig_exp4_residual_hist_{safe_split}_{safe_model}.png"
        )
        save_scatter(
            sub.sample(min(len(sub), 1000), random_state=42),
            f"True vs predicted: {split_label} / {model_label}",
            fig_dir / f"fig_exp4_true_vs_pred_{safe_split}_{safe_model}.png"
        )

    # Figure 5: dimension-level bias summary.
    if dim_path.exists():
        dim = pd.read_csv(dim_path)
        dim_main = dim[
            dim["shot"].eq(args.shot) &
            dim["calibration"].eq(args.calibration) &
            dim["split_file"].isin(ROBUSTNESS_SPLITS)
        ].copy()
        if models != ["all"]:
            dim_main = dim_main[dim_main["model_label"].isin(models)]

        # For each model and dimension, aggregate absolute calibrated bias.
        if not dim_main.empty:
            agg = (
                dim_main.groupby(["dimension", "model_label"])
                .agg(mean_abs_bias=("abs_bias", "mean"), mean_MAE=("MAE", "mean"))
                .reset_index()
            )
            for dim_name, sub in agg.groupby("dimension"):
                sub = sub.sort_values("mean_abs_bias")
                save_bar(
                    sub,
                    "model_label",
                    "mean_abs_bias",
                    f"Mean absolute bias by {dim_name}",
                    "Mean absolute bias",
                    fig_dir / f"fig_exp4_dimension_abs_bias_{dim_name}.png"
                )

    # Subject-level improvement distribution.
    if subj_path.exists():
        subj = pd.read_csv(subj_path)
        subj_main = subj[
            subj["shot"].eq(args.shot) &
            subj["calibration"].eq(args.calibration) &
            subj["split_file"].isin(ROBUSTNESS_SPLITS)
        ].copy()
        if models != ["all"]:
            subj_main = subj_main[subj_main["model_label"].isin(models)]
        if not subj_main.empty:
            for split_label, sub in subj_main.groupby("split_label"):
                plt.figure(figsize=(8, 5))
                for model, ss in sub.groupby("model_label"):
                    vals = ss["RMSE_reduction"].dropna().to_numpy()
                    if len(vals) == 0:
                        continue
                    plt.hist(vals, bins=20, alpha=0.5, label=model)
                plt.axvline(0, linestyle="--")
                plt.xlabel("Subject-level RMSE reduction")
                plt.ylabel("Number of subjects")
                plt.title(f"Subject-level improvement distribution: {split_label}")
                plt.legend(fontsize=8)
                plt.tight_layout()
                safe = str(split_label).replace("→", "to").replace(" ", "_").lower()
                plt.savefig(fig_dir / f"fig_exp4_subject_rmse_reduction_{safe}.png", dpi=300)
                plt.close()

    print(f"[OK] Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()
