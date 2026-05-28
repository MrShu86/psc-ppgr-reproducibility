import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meter_ppgr.io_utils import load_config
from meter_ppgr.baselines import run_tabular_baselines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_tabular_baselines(config)


if __name__ == "__main__":
    main()
