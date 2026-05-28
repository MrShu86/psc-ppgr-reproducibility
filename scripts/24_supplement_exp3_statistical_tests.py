import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception as e:
    stats = None
    print(f"[WARN] scipy not available: {e}")


ROBUSTNESS_SPLITS = [
    "cross_device_dexcom_to_libre.csv",
    "cross_device_libre_to_dexcom.csv",
    "cross_subject_split.csv",
    "cross_setting_mealtype_holdout_breakfast.csv",
]

LABEL_MAP = {
    "random_meal_split.csv": "Random",
    "cross_subject_split.csv": "Cross-subject",
    "cross_device_dexcom_to_libre.csv": "Dexcom→Libre",
    "cross_device_libre_to_dexcom.csv": "Libre→Dexcom",
    "cross_setting_mealtype_holdout_breakfast.csv": "Breakfast holdout",
}


def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    alpha = 1 - ci
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def cohen_d_paired(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return np.nan
    return float(np.mean(diff) / sd)


def sign_test_p(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    if len(diff) == 0:
        return np.nan
    n_pos = int(np.sum(diff > 0))
    n = len(diff)
    if stats is None:
        return np.nan
    try:
        return float(stats.binomtest(n_pos, n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def paired_test(diff, alternative="two-sided"):
    """
    diff > 0 means the tested method improves over baseline.
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    out = {
        "n_pairs": n,
        "mean_delta_RMSE": float(np.mean(diff)) if n else np.nan,
        "median_delta_RMSE": float(np.median(diff)) if n else np.nan,
        "std_delta_RMSE": float(np.std(diff, ddof=1)) if n > 1 else np.nan,
        "cohen_d_paired": cohen_d_paired(diff),
        "n_improved": int(np.sum(diff > 0)) if n else 0,
        "n_worse": int(np.sum(diff < 0)) if n else 0,
        "improvement_rate_%": float(np.mean(diff > 0) * 100) if n else np.nan,
        "bootstrap_mean_delta_CI95_low": np.nan,
        "bootstrap_mean_delta_CI95_high": np.nan,
        "paired_t_p": np.nan,
        "wilcoxon_p": np.nan,
        "sign_test_p": sign_test_p(diff),
    }
    if n >= 2:
        low, high = bootstrap_ci(diff)
        out["bootstrap_mean_delta_CI95_low"] = low
        out["bootstrap_mean_delta_CI95_high"] = high

    if stats is not None and n >= 2:
        try:
            out["paired_t_p"] = float(stats.ttest_1samp(diff, popmean=0.0, alternative=alternative).pvalue)
        except Exception:
            pass

        # Wilcoxon requires at least one non-zero diff.
        if np.any(diff != 0):
            try:
                out["wilcoxon_p"] = float(stats.wilcoxon(diff, alternative=alternative, zero_method="wilcox").pvalue)
            except Exception:
                pass

    return out


def summarize_claim(row, alpha=0.05):
    """
    Conservative human-readable claim.
    """
    mean_delta = row.get("mean_delta_RMSE", np.nan)
    ci_low = row.get("bootstrap_mean_delta_CI95_low", np.nan)
    p_w = row.get("wilcoxon_p", np.nan)
    n = row.get("n_pairs", 0)

    if not np.isfinite(mean_delta) or n < 5:
        return "insufficient_subject_pairs"
    if mean_delta > 0 and np.isfinite(ci_low) and ci_low > 0 and np.isfinite(p_w) and p_w < alpha:
        return "significant_improvement"
    if mean_delta > 0 and (not np.isfinite(p_w) or p_w >= alpha):
        return "positive_trend_not_significant"
    if mean_delta <= 0 and np.isfinite(p_w) and p_w < alpha:
        return "significant_degradation"
    return "no_clear_effect"


def load_detail(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"method", "encoder", "split_file", "fold_subject", "shot", "personalization", "RMSE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df["shot"] = pd.to_numeric(df["shot"], errors="coerce").astype("Int64")
    return df


def test_support_vs_no_update(detail: pd.DataFrame):
    """
    For each method/encoder/split/shot, compare support calibration against
    the same model's global_no_update on exactly the same support/query protocol.
    Positive delta = global_no_update RMSE - calibrated RMSE.
    """
    rows = []
    calib_modes = ["support_residual_calibration", "support_affine_calibration"]

    keys = ["method", "encoder", "split_file", "shot"]
    for key_vals, sub in detail.groupby(keys, dropna=False):
        base = sub[sub["personalization"].isin(["global_no_update", "global_0shot"])].copy()
        if base.empty:
            continue

        # For shot=0 only global_0shot exists; support calibration does not.
        shot = key_vals[-1]
        if int(shot) == 0:
            continue

        base = base[base["personalization"].eq("global_no_update")]
        if base.empty:
            continue

        for mode in calib_modes:
            cal = sub[sub["personalization"].eq(mode)].copy()
            if cal.empty:
                continue

            merged = base[["fold_subject", "RMSE"]].merge(
                cal[["fold_subject", "RMSE"]],
                on="fold_subject",
                suffixes=("_no_update", "_calibrated")
            )
            if merged.empty:
                continue

            diff = merged["RMSE_no_update"].to_numpy(dtype=float) - merged["RMSE_calibrated"].to_numpy(dtype=float)
            test = paired_test(diff)

            row = dict(zip(keys, key_vals))
            row.update({
                "split_label": LABEL_MAP.get(row["split_file"], row["split_file"]),
                "comparison": f"{mode} vs global_no_update",
                "baseline": "global_no_update",
                "tested": mode,
                "baseline_RMSE_mean": float(merged["RMSE_no_update"].mean()),
                "tested_RMSE_mean": float(merged["RMSE_calibrated"].mean()),
                "relative_RMSE_reduction_%": float((merged["RMSE_no_update"].mean() - merged["RMSE_calibrated"].mean()) / merged["RMSE_no_update"].mean() * 100),
            })
            row.update(test)
            row["claim"] = summarize_claim(row)
            rows.append(row)

    return pd.DataFrame(rows)


def test_meter_psc_vs_tabular_fixed5(detail: pd.DataFrame, psc_method="METER-v1", psc_encoder="tcn", shot=5, calibration="support_residual_calibration"):
    """
    Fair fixed setting:
    METER-PSC = METER-v1-tcn + 5-shot residual
    compared with each Tabular-* model under the same shot and calibration.
    Positive delta = tabular RMSE - METER-PSC RMSE, so positive favors METER-PSC.
    """
    rows = []
    psc = detail[
        (detail["method"].eq(psc_method)) &
        (detail["encoder"].eq(psc_encoder)) &
        (detail["shot"].astype(int).eq(shot)) &
        (detail["personalization"].eq(calibration))
    ].copy()

    if psc.empty:
        return pd.DataFrame()

    tab = detail[
        (detail["method"].astype(str).str.startswith("Tabular-", na=False)) &
        (detail["shot"].astype(int).eq(shot)) &
        (detail["personalization"].eq(calibration))
    ].copy()

    for split_file, psc_sub in psc.groupby("split_file"):
        tab_split = tab[tab["split_file"].eq(split_file)].copy()
        if tab_split.empty:
            continue

        # Each tabular method vs METER-PSC.
        for (tab_method, tab_encoder), tab_sub in tab_split.groupby(["method", "encoder"]):
            merged = tab_sub[["fold_subject", "RMSE"]].merge(
                psc_sub[["fold_subject", "RMSE"]],
                on="fold_subject",
                suffixes=("_tabular", "_psc")
            )
            if merged.empty:
                continue

            diff = merged["RMSE_tabular"].to_numpy(dtype=float) - merged["RMSE_psc"].to_numpy(dtype=float)
            test = paired_test(diff)

            row = {
                "split_file": split_file,
                "split_label": LABEL_MAP.get(split_file, split_file),
                "comparison": f"METER-PSC vs {tab_method} fixed {shot}-shot residual",
                "baseline": tab_method,
                "tested": f"{psc_method}-{psc_encoder}+{shot}shot_residual",
                "shot": shot,
                "personalization": calibration,
                "tabular_RMSE_mean": float(merged["RMSE_tabular"].mean()),
                "METER_PSC_RMSE_mean": float(merged["RMSE_psc"].mean()),
                "relative_RMSE_reduction_by_PSC_%": float((merged["RMSE_tabular"].mean() - merged["RMSE_psc"].mean()) / merged["RMSE_tabular"].mean() * 100),
            }
            row.update(test)
            row["claim"] = summarize_claim(row)
            rows.append(row)

        # Best tabular fixed 5-shot residual per split based on subject-mean RMSE.
        best_candidates = []
        for (tab_method, tab_encoder), tab_sub in tab_split.groupby(["method", "encoder"]):
            merged = tab_sub[["fold_subject", "RMSE"]].merge(
                psc_sub[["fold_subject", "RMSE"]],
                on="fold_subject",
                suffixes=("_tabular", "_psc")
            )
            if not merged.empty:
                best_candidates.append((tab_method, tab_encoder, merged["RMSE_tabular"].mean(), merged))
        if best_candidates:
            best_method, best_encoder, best_rmse, merged = sorted(best_candidates, key=lambda x: x[2])[0]
            diff = merged["RMSE_tabular"].to_numpy(dtype=float) - merged["RMSE_psc"].to_numpy(dtype=float)
            test = paired_test(diff)
            row = {
                "split_file": split_file,
                "split_label": LABEL_MAP.get(split_file, split_file),
                "comparison": f"METER-PSC vs BEST_TABULAR fixed {shot}-shot residual",
                "baseline": best_method,
                "tested": f"{psc_method}-{psc_encoder}+{shot}shot_residual",
                "shot": shot,
                "personalization": calibration,
                "tabular_RMSE_mean": float(merged["RMSE_tabular"].mean()),
                "METER_PSC_RMSE_mean": float(merged["RMSE_psc"].mean()),
                "relative_RMSE_reduction_by_PSC_%": float((merged["RMSE_tabular"].mean() - merged["RMSE_psc"].mean()) / merged["RMSE_tabular"].mean() * 100),
            }
            row.update(test)
            row["claim"] = summarize_claim(row)
            rows.append(row)

    return pd.DataFrame(rows)


def test_best_tabular_support_vs_no_update(detail: pd.DataFrame):
    """
    For each tabular method/split, test its calibrated result vs its no-update result.
    This quantifies whether support calibration significantly helps strong baselines.
    """
    tab = detail[detail["method"].astype(str).str.startswith("Tabular-", na=False)].copy()
    if tab.empty:
        return pd.DataFrame()
    return test_support_vs_no_update(tab)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", default="outputs/supplement_exp2/supplement_exp2_support_all_detail.csv")
    parser.add_argument("--output_dir", default="outputs/supplement_exp3")
    parser.add_argument("--psc_method", default="METER-v1")
    parser.add_argument("--psc_encoder", default="tcn")
    parser.add_argument("--psc_shot", type=int, default=5)
    parser.add_argument("--psc_calibration", default="support_residual_calibration")
    args = parser.parse_args()

    detail_path = Path(args.detail)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail = load_detail(detail_path)

    support_tests = test_support_vs_no_update(detail)
    support_tests.to_csv(output_dir / "supplement_exp3_support_vs_no_update_tests.csv", index=False)

    psc_vs_tabular = test_meter_psc_vs_tabular_fixed5(
        detail,
        psc_method=args.psc_method,
        psc_encoder=args.psc_encoder,
        shot=args.psc_shot,
        calibration=args.psc_calibration
    )
    psc_vs_tabular.to_csv(output_dir / "supplement_exp3_meter_psc_vs_tabular_fixed5_tests.csv", index=False)

    tabular_support_tests = test_best_tabular_support_vs_no_update(detail)
    tabular_support_tests.to_csv(output_dir / "supplement_exp3_tabular_support_vs_no_update_tests.csv", index=False)

    # Recommended claims summary.
    claim_rows = []
    if not support_tests.empty:
        main_support = support_tests[
            (support_tests["split_file"].isin(ROBUSTNESS_SPLITS)) &
            (support_tests["shot"].astype(int).isin([5, 10])) &
            (support_tests["tested"].isin(["support_residual_calibration", "support_affine_calibration"]))
        ].copy()
        for _, r in main_support.iterrows():
            if r["claim"] in ["significant_improvement", "positive_trend_not_significant"]:
                claim_rows.append({
                    "category": "support_calibration_effect",
                    "split_label": r["split_label"],
                    "method": f"{r['method']}-{r['encoder']}",
                    "shot": r["shot"],
                    "comparison": r["comparison"],
                    "mean_delta_RMSE": r["mean_delta_RMSE"],
                    "relative_RMSE_reduction_%": r["relative_RMSE_reduction_%"],
                    "wilcoxon_p": r["wilcoxon_p"],
                    "claim": r["claim"],
                })

    if not psc_vs_tabular.empty:
        main_psc = psc_vs_tabular[psc_vs_tabular["comparison"].str.contains("BEST_TABULAR", na=False)].copy()
        for _, r in main_psc.iterrows():
            claim_rows.append({
                "category": "meter_psc_vs_best_tabular_fixed5",
                "split_label": r["split_label"],
                "method": r["tested"],
                "shot": r["shot"],
                "comparison": r["comparison"],
                "mean_delta_RMSE": r["mean_delta_RMSE"],
                "relative_RMSE_reduction_%": r["relative_RMSE_reduction_by_PSC_%"],
                "wilcoxon_p": r["wilcoxon_p"],
                "claim": r["claim"],
            })

    claims = pd.DataFrame(claim_rows)
    claims.to_csv(output_dir / "supplement_exp3_recommended_claims.csv", index=False)

    # Excel workbook.
    try:
        with pd.ExcelWriter(output_dir / "supplement_exp3_statistical_tests.xlsx") as writer:
            support_tests.to_excel(writer, sheet_name="support_vs_no_update", index=False)
            psc_vs_tabular.to_excel(writer, sheet_name="PSC_vs_tabular_fixed5", index=False)
            tabular_support_tests.to_excel(writer, sheet_name="tabular_support_tests", index=False)
            claims.to_excel(writer, sheet_name="recommended_claims", index=False)
    except Exception as e:
        print(f"[WARN] Failed to write xlsx: {e}")

    print(f"[OK] Saved: {output_dir / 'supplement_exp3_support_vs_no_update_tests.csv'}")
    print(f"[OK] Saved: {output_dir / 'supplement_exp3_meter_psc_vs_tabular_fixed5_tests.csv'}")
    print(f"[OK] Saved: {output_dir / 'supplement_exp3_tabular_support_vs_no_update_tests.csv'}")
    print(f"[OK] Saved: {output_dir / 'supplement_exp3_recommended_claims.csv'}")

    if not claims.empty:
        print("\n[RECOMMENDED CLAIMS PREVIEW]")
        cols = ["category", "split_label", "method", "shot", "relative_RMSE_reduction_%", "wilcoxon_p", "claim"]
        print(claims[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
