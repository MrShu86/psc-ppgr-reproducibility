import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--target", default="iauc_2h")
    parser.add_argument("--encoder", default="tcn")
    args = parser.parse_args()

    out = Path(args.output_dir)
    res = out / "results"
    summary_dir = out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(res.glob(f"meter_v2_{args.encoder}_*_{args.target}.csv"))
    files = [p for p in files if not any(s in p.name for s in ["predictions", "support", "repr"])]
    if files:
        df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        df.to_csv(summary_dir / f"meter_v2_{args.encoder}_main_results_{args.target}.csv", index=False)
        print(f"[OK] Saved {summary_dir / f'meter_v2_{args.encoder}_main_results_{args.target}.csv'}")

    support_files = sorted(res.glob(f"meter_v2_{args.encoder}_*_{args.target}_support_summary.csv"))
    if support_files:
        sup = pd.concat([pd.read_csv(p) for p in support_files], ignore_index=True)
        sup.to_csv(summary_dir / f"meter_v2_{args.encoder}_support_summary_{args.target}.csv", index=False)
        print(f"[OK] Saved {summary_dir / f'meter_v2_{args.encoder}_support_summary_{args.target}.csv'}")

    # Also merge v1 and sequence results if present.
    seq_files = sorted(res.glob(f"sequence_baseline_*_{args.target}.csv"))
    seq_files = [p for p in seq_files if "predictions" not in p.name]
    if seq_files:
        seq = pd.concat([pd.read_csv(p) for p in seq_files], ignore_index=True)
        seq.to_csv(summary_dir / f"sequence_baseline_all_{args.target}.csv", index=False)
        print(f"[OK] Saved {summary_dir / f'sequence_baseline_all_{args.target}.csv'}")

    v1_files = sorted(res.glob(f"meter_v1_{args.encoder}_*_{args.target}.csv"))
    v1_files = [p for p in v1_files if not any(s in p.name for s in ["predictions", "repr"])]
    if v1_files:
        v1 = pd.concat([pd.read_csv(p) for p in v1_files], ignore_index=True)
        v1.to_csv(summary_dir / f"meter_v1_{args.encoder}_all_{args.target}.csv", index=False)
        print(f"[OK] Saved {summary_dir / f'meter_v1_{args.encoder}_all_{args.target}.csv'}")


if __name__ == "__main__":
    main()
