import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meter_ppgr.io_utils import load_config
from meter_ppgr.splits import make_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--clean_results", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out = Path(config["output_dir"])
    split_dir = out / "splits"
    result_dir = out / "results"

    if split_dir.exists():
        for p in split_dir.glob("*.csv"):
            p.unlink()
        print(f"[OK] Removed old split CSV files from {split_dir}")

    if args.clean_results and result_dir.exists():
        for p in result_dir.glob("tabular_baselines_*.csv"):
            p.unlink()
        print(f"[OK] Removed old tabular baseline CSV files from {result_dir}")

    make_splits(config)
    print("[OK] Rebuilt split files.")


if __name__ == "__main__":
    main()
