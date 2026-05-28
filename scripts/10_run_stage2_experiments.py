import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("\n[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--skip_sequence", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    run([py, str(ROOT / "scripts" / "05_normalize_meal_types.py"), "--config", args.config, "--overwrite"])
    run([py, str(ROOT / "scripts" / "11_clean_stale_splits.py"), "--config", args.config, "--clean_results"])
    run([py, str(ROOT / "scripts" / "04_run_tabular_baselines.py"), "--config", args.config])
    run([py, str(ROOT / "scripts" / "07_run_loso_tabular.py"), "--config", args.config])
    run([py, str(ROOT / "scripts" / "08_run_cold_start_tabular.py"), "--config", args.config])

    if not args.skip_sequence:
        jobs = [
            ("random_meal_split.csv", "gru"),
            ("cross_subject_split.csv", "gru"),
            ("cross_device_libre_to_dexcom.csv", "gru"),
            ("cross_device_dexcom_to_libre.csv", "gru"),
            ("cross_setting_mealtype_holdout_breakfast.csv", "gru"),
            ("random_meal_split.csv", "tcn"),
            ("cross_subject_split.csv", "tcn"),
        ]
        for split, model in jobs:
            run([
                py, str(ROOT / "scripts" / "09_run_premeal_sequence_baseline.py"),
                "--config", args.config,
                "--split", split,
                "--model", model,
            ])


if __name__ == "__main__":
    main()
