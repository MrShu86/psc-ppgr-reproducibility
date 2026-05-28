import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meter_ppgr.io_utils import load_config
from meter_ppgr.meter_v1 import run_meter_v1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="random_meal_split.csv")
    parser.add_argument("--encoder", default="tcn", choices=["tcn", "gru"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lambda_device", type=float, default=0.10)
    parser.add_argument("--no_device_adv", action="store_true")
    parser.add_argument("--no_adapter", action="store_true")
    parser.add_argument("--include_device_as_predictor", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    run_meter_v1(
        config=config,
        split_name=args.split,
        encoder=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        lambda_device=args.lambda_device,
        use_device_adv=not args.no_device_adv,
        use_adapter=not args.no_adapter,
        include_device_as_predictor=args.include_device_as_predictor,
    )


if __name__ == "__main__":
    main()
