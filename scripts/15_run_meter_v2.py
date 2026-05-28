import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meter_ppgr.io_utils import load_config
from meter_ppgr.meter_v2 import run_meter_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="random_meal_split.csv")
    parser.add_argument("--encoder", default="tcn", choices=["tcn", "gru"])
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lambda_device", type=float, default=0.10)
    parser.add_argument("--lambda_pair_z", type=float, default=0.05)
    parser.add_argument("--lambda_pair_pred", type=float, default=0.05)
    parser.add_argument("--no_device_adv", action="store_true")
    parser.add_argument("--no_pair_consistency", action="store_true")
    parser.add_argument("--no_calibration_head", action="store_true")
    parser.add_argument("--include_device_as_predictor", action="store_true")
    parser.add_argument("--shots", default="0,1,3,5,10")
    args = parser.parse_args()

    config = load_config(args.config)
    shots = tuple(int(x.strip()) for x in args.shots.split(",") if x.strip())

    run_meter_v2(
        config=config,
        split_name=args.split,
        encoder=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        lambda_device=args.lambda_device,
        lambda_pair_z=args.lambda_pair_z,
        lambda_pair_pred=args.lambda_pair_pred,
        use_device_adv=not args.no_device_adv,
        use_pair_consistency=not args.no_pair_consistency,
        use_calibration_head=not args.no_calibration_head,
        include_device_as_predictor=args.include_device_as_predictor,
        shots=shots,
    )


if __name__ == "__main__":
    main()
