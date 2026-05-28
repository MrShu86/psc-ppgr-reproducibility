import argparse
from pathlib import Path
import sys
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meter_ppgr.io_utils import load_config
from meter_ppgr.stage2_utils import normalize_meal_type_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out = Path(config["output_dir"])
    meta_path = out / "events_metadata.csv"

    meta = pd.read_csv(meta_path)
    before = meta["meal_type"].value_counts(dropna=False).rename_axis("meal_type_raw").reset_index(name="count")

    meta["meal_type_raw"] = meta["meal_type"]
    meta["meal_type"] = meta["meal_type"].map(normalize_meal_type_value)
    meta["setting_meal_type"] = meta["meal_type"]

    after = meta["meal_type"].value_counts(dropna=False).rename_axis("meal_type_normalized").reset_index(name="count")

    before.to_csv(out / "meal_type_counts_before_normalization.csv", index=False)
    after.to_csv(out / "meal_type_counts_after_normalization.csv", index=False)
    meta.to_csv(out / "events_metadata_mealtype_normalized.csv", index=False)

    print("[MEAL TYPE] Before:")
    print(before.to_string(index=False))
    print("\n[MEAL TYPE] After:")
    print(after.to_string(index=False))

    if args.overwrite:
        backup = out / "events_metadata_before_mealtype_normalization.csv"
        if not backup.exists():
            shutil.copy2(meta_path, backup)
        meta.to_csv(meta_path, index=False)
        print(f"[OK] Backed up original to {backup}")
        print(f"[OK] Overwrote {meta_path}")


if __name__ == "__main__":
    main()
