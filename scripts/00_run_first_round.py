import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    "01_build_meal_events.py",
    "02_profile_dataset.py",
    "03_make_splits.py",
    "04_run_tabular_baselines.py",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()

    for step in STEPS:
        cmd = [sys.executable, str(ROOT / "scripts" / step), "--config", args.config]
        print("\n[RUN]", " ".join(cmd))
        subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
