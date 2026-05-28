import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    out = Path(args.output_dir)
    res_dir = out / "results"
    summary_dir = out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    tab_path = res_dir / "tabular_baselines_iauc_2h.csv"
    if tab_path.exists():
        tab = pd.read_csv(tab_path)
        tab = tab[~tab["split_file"].str.contains("holdout_snacks", case=False, na=False)].copy()
        best = tab.loc[tab.groupby("split_file")["RMSE"].idxmin()].sort_values("RMSE")
        best.to_csv(summary_dir / "best_tabular_result_per_split.csv", index=False)
        print(f"[OK] Saved {summary_dir / 'best_tabular_result_per_split.csv'}")

    loso_path = res_dir / "loso_tabular_summary_iauc_2h.csv"
    if loso_path.exists():
        loso = pd.read_csv(loso_path).sort_values("RMSE_mean")
        loso.to_csv(summary_dir / "loso_summary_sorted.csv", index=False)
        print(f"[OK] Saved {summary_dir / 'loso_summary_sorted.csv'}")

    cold_path = res_dir / "cold_start_tabular_summary_iauc_2h.csv"
    if cold_path.exists():
        cold = pd.read_csv(cold_path)
        cold.to_csv(summary_dir / "cold_start_summary.csv", index=False)
        print(f"[OK] Saved {summary_dir / 'cold_start_summary.csv'}")

    seq_files = sorted(res_dir.glob("sequence_baseline_*.csv"))
    if seq_files:
        seq = pd.concat([pd.read_csv(p) for p in seq_files], ignore_index=True)
        seq.to_csv(summary_dir / "sequence_baseline_summary.csv", index=False)
        print(f"[OK] Saved {summary_dir / 'sequence_baseline_summary.csv'}")


if __name__ == "__main__":
    main()
