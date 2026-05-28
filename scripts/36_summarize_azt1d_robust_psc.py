import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robust_dir", default="outputs_azt1d/robust_psc_check")
    parser.add_argument("--output_dir", default="outputs_azt1d/robust_psc_check/figures")
    args = parser.parse_args()

    robust_dir = Path(args.robust_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(robust_dir / "azt1d_robust_psc_summary.csv")
    comp_path = robust_dir / "azt1d_robust_psc_best_calibrated_vs_no_update.csv"
    tests_path = robust_dir / "azt1d_robust_psc_paired_tests.csv"
    comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
    tests = pd.read_csv(tests_path) if tests_path.exists() else pd.DataFrame()

    # 1. Best calibrated vs no-update curve per model.
    if not comp.empty:
        plt.figure(figsize=(9, 5))
        for model, sub in comp.groupby("model"):
            sub = sub.sort_values("shot")
            plt.plot(sub["shot"], sub["relative_RMSE_reduction_%"], marker="o", label=model)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Number of support events")
        plt.ylabel("Best calibrated RMSE reduction vs no-update (%)")
        plt.title("AZT1D robust PSC sensitivity: best calibrated improvement")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / "fig_azt1d_robust_psc_best_improvement_curve.png", dpi=300)
        plt.close()

        # best calibration labels table-like bar for 10/20/30 shots.
        sub = comp[comp["shot"].isin([5, 10, 20, 30])].copy()
        sub["plot_label"] = sub["model"] + "\n" + sub["shot"].astype(str) + "-shot"
        plt.figure(figsize=(10, 5))
        x = np.arange(len(sub))
        plt.bar(x, sub["relative_RMSE_reduction_%"])
        plt.axhline(0, linestyle="--")
        plt.xticks(x, sub["plot_label"], rotation=35, ha="right")
        plt.ylabel("RMSE reduction vs no-update (%)")
        plt.title("AZT1D robust PSC: selected support sizes")
        plt.tight_layout()
        plt.savefig(output_dir / "fig_azt1d_robust_psc_selected_shots_bar.png", dpi=300)
        plt.close()

    # 2. RMSE curves for core calibrations.
    core = [
        "global_no_update", "global_0shot",
        "mean_shrink_tau10", "median", "trimmed",
        "mealtype_mean_shrink_tau10", "baselinebin_mean_shrink_tau10",
        "mean_clipped_clip1000"
    ]
    plot_df = summary[summary["personalization"].isin(core)].copy()
    for model, subm in plot_df.groupby("model"):
        plt.figure(figsize=(9, 5))
        for pers, sub in subm.groupby("personalization"):
            sub = sub.sort_values("shot")
            plt.plot(sub["shot"], sub["RMSE_mean"], marker="o", label=pers)
        plt.xlabel("Number of support events")
        plt.ylabel("Subject-level mean RMSE")
        plt.title(f"AZT1D robust PSC RMSE curve: {model}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        safe = str(model).replace(" ", "_")
        plt.savefig(output_dir / f"fig_azt1d_robust_psc_rmse_curve_{safe}.png", dpi=300)
        plt.close()

    # 3. Heatmap-like table by model / calibration for 10-shot.
    for shot in [5, 10, 20, 30]:
        ss = summary[summary["shot"].eq(shot)].copy()
        if ss.empty:
            continue
        pivot = ss.pivot_table(index="personalization", columns="model", values="RMSE_mean", aggfunc="min")
        pivot.to_csv(robust_dir / f"azt1d_robust_psc_rmse_pivot_{shot}shot.csv")

        # Plot top 15 calibrations by mean RMSE.
        ss2 = ss[~ss["personalization"].isin(["global_no_update", "global_0shot"])].copy()
        if ss2.empty:
            continue
        avg = ss2.groupby("personalization")["RMSE_mean"].mean().sort_values().head(15).reset_index()
        plt.figure(figsize=(9, 6))
        x = np.arange(len(avg))
        plt.barh(avg["personalization"], avg["RMSE_mean"])
        plt.xlabel("Mean RMSE across models")
        plt.title(f"Top robust PSC variants on AZT1D ({shot}-shot)")
        plt.tight_layout()
        plt.savefig(output_dir / f"fig_azt1d_robust_psc_top_variants_{shot}shot.png", dpi=300)
        plt.close()

    # 4. P-value vs effect plot.
    if not tests.empty:
        t = tests.copy()
        t = t[np.isfinite(t["wilcoxon_p"])]
        if not t.empty:
            t["neglog10_p"] = -np.log10(t["wilcoxon_p"].clip(lower=1e-300))
            plt.figure(figsize=(8, 5))
            plt.scatter(t["relative_RMSE_reduction_%"], t["neglog10_p"], alpha=0.6)
            plt.axvline(0, linestyle="--")
            plt.axhline(-np.log10(0.05), linestyle="--")
            plt.xlabel("RMSE reduction vs no-update (%)")
            plt.ylabel("-log10 Wilcoxon p")
            plt.title("AZT1D robust PSC effect vs significance")
            plt.tight_layout()
            plt.savefig(output_dir / "fig_azt1d_robust_psc_effect_vs_pvalue.png", dpi=300)
            plt.close()

    print(f"[OK] Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
