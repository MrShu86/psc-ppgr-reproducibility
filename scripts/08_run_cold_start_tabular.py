import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meter_ppgr.io_utils import load_config
from meter_ppgr.cold_start import run_cold_start_tabular


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--shots", default="0,1,3,5,10")
    args = parser.parse_args()
    config = load_config(args.config)
    shots = tuple(int(x.strip()) for x in args.shots.split(",") if x.strip())
    run_cold_start_tabular(config, shots=shots)


if __name__ == "__main__":
    main()
