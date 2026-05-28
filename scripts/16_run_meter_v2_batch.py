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
    parser.add_argument("--encoder", default="tcn", choices=["tcn", "gru"])
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--with_ablations", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    splits = [
        "random_meal_split.csv",
        "cross_subject_split.csv",
        "cross_device_libre_to_dexcom.csv",
        "cross_device_dexcom_to_libre.csv",
        "cross_setting_mealtype_holdout_breakfast.csv",
    ]

    for split in splits:
        run([
            py, str(ROOT / "scripts" / "15_run_meter_v2.py"),
            "--config", args.config,
            "--split", split,
            "--encoder", args.encoder,
            "--epochs", str(args.epochs),
        ])

    if args.with_ablations:
        for split in ["random_meal_split.csv", "cross_subject_split.csv", "cross_setting_mealtype_holdout_breakfast.csv"]:
            run([
                py, str(ROOT / "scripts" / "15_run_meter_v2.py"),
                "--config", args.config,
                "--split", split,
                "--encoder", args.encoder,
                "--epochs", str(args.epochs),
                "--no_pair_consistency",
            ])
            run([
                py, str(ROOT / "scripts" / "15_run_meter_v2.py"),
                "--config", args.config,
                "--split", split,
                "--encoder", args.encoder,
                "--epochs", str(args.epochs),
                "--no_calibration_head",
            ])

if __name__ == "__main__":
    main()
