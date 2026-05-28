import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meter_ppgr.io_utils import load_config
from meter_ppgr.sequence_baselines import run_sequence_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="random_meal_split.csv")
    parser.add_argument("--model", default="gru", choices=["mlp", "gru", "tcn", "transformer"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    run_sequence_baseline(
        config=config,
        split_name=args.split,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
