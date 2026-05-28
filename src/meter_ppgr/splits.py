from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .io_utils import ensure_dir


def _assign_split(meta: pd.DataFrame, train_idx, val_idx, test_idx) -> pd.DataFrame:
    out = meta[["event_id", "paired_event_id", "subject_id", "device", "meal_type"]].copy()
    out["split"] = "unused"
    out.loc[train_idx, "split"] = "train"
    out.loc[val_idx, "split"] = "val"
    out.loc[test_idx, "split"] = "test"
    return out


def _paired_group_split(meta: pd.DataFrame, test_size: float, val_size: float, seed: int):
    groups = meta["paired_event_id"].drop_duplicates().to_numpy()
    trainval_groups, test_groups = train_test_split(groups, test_size=test_size, random_state=seed)
    rel_val = val_size / max(1e-8, 1 - test_size)
    train_groups, val_groups = train_test_split(trainval_groups, test_size=rel_val, random_state=seed)

    train_idx = meta.index[meta["paired_event_id"].isin(train_groups)]
    val_idx = meta.index[meta["paired_event_id"].isin(val_groups)]
    test_idx = meta.index[meta["paired_event_id"].isin(test_groups)]
    return train_idx, val_idx, test_idx


def make_splits(config: dict) -> None:
    out = Path(config["output_dir"])
    split_dir = ensure_dir(out / "splits")
    meta_path = out / "events_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError("Run 01_build_meal_events.py first.")

    meta = pd.read_csv(meta_path)
    seed = int(config.get("random_seed", 42))
    test_size = float(config.get("test_size", 0.20))
    val_size = float(config.get("val_size", 0.10))

    # 1) Random meal split, grouped by paired_event_id to avoid same meal leakage across devices.
    train_idx, val_idx, test_idx = _paired_group_split(meta, test_size, val_size, seed)
    _assign_split(meta, train_idx, val_idx, test_idx).to_csv(split_dir / "random_meal_split.csv", index=False)

    # 2) Cross-device splits.
    devices = sorted(meta["device"].dropna().unique().tolist())
    for src in devices:
        for tgt in devices:
            if src == tgt:
                continue
            split = meta[["event_id", "paired_event_id", "subject_id", "device", "meal_type"]].copy()
            split["split"] = "unused"
            source_idx = meta.index[meta["device"] == src]
            target_idx = meta.index[meta["device"] == tgt]

            # Validation: subset of source-domain paired events.
            source_pairs = meta.loc[source_idx, "paired_event_id"].drop_duplicates().to_numpy()
            if len(source_pairs) >= 5:
                train_pairs, val_pairs = train_test_split(source_pairs, test_size=0.15, random_state=seed)
                split.loc[meta.index[(meta["device"] == src) & (meta["paired_event_id"].isin(train_pairs))], "split"] = "train"
                split.loc[meta.index[(meta["device"] == src) & (meta["paired_event_id"].isin(val_pairs))], "split"] = "val"
            else:
                split.loc[source_idx, "split"] = "train"

            split.loc[target_idx, "split"] = "test"
            split.to_csv(split_dir / f"cross_device_{src}_to_{tgt}.csv", index=False)

    # 3) Cross-subject split.
    subjects = meta["subject_id"].drop_duplicates().to_numpy()
    trainval_subj, test_subj = train_test_split(subjects, test_size=test_size, random_state=seed)
    rel_val = val_size / max(1e-8, 1 - test_size)
    train_subj, val_subj = train_test_split(trainval_subj, test_size=rel_val, random_state=seed)
    split = meta[["event_id", "paired_event_id", "subject_id", "device", "meal_type"]].copy()
    split["split"] = "unused"
    split.loc[meta["subject_id"].isin(train_subj), "split"] = "train"
    split.loc[meta["subject_id"].isin(val_subj), "split"] = "val"
    split.loc[meta["subject_id"].isin(test_subj), "split"] = "test"
    split.to_csv(split_dir / "cross_subject_split.csv", index=False)

    # 4) LOSO index file.
    loso_rows = []
    for s in subjects:
        for eid, subj in zip(meta["event_id"], meta["subject_id"]):
            loso_rows.append({
                "fold_subject": s,
                "event_id": eid,
                "split": "test" if subj == s else "train",
            })
    pd.DataFrame(loso_rows).to_csv(split_dir / "loso_splits.csv", index=False)

    # 5) Cross-setting by meal type.
    if "meal_type" in meta.columns:
        meal_types = sorted([x for x in meta["meal_type"].dropna().unique().tolist() if str(x).strip()])
        for holdout in meal_types:
            split = meta[["event_id", "paired_event_id", "subject_id", "device", "meal_type"]].copy()
            split["split"] = "unused"
            test_mask = meta["meal_type"] == holdout
            trainval_pairs = meta.loc[~test_mask, "paired_event_id"].drop_duplicates().to_numpy()
            if len(trainval_pairs) >= 5:
                train_pairs, val_pairs = train_test_split(trainval_pairs, test_size=0.15, random_state=seed)
                split.loc[(~test_mask) & (meta["paired_event_id"].isin(train_pairs)), "split"] = "train"
                split.loc[(~test_mask) & (meta["paired_event_id"].isin(val_pairs)), "split"] = "val"
            else:
                split.loc[~test_mask, "split"] = "train"
            split.loc[test_mask, "split"] = "test"
            safe_name = str(holdout).lower().replace(" ", "_").replace("/", "_")
            split.to_csv(split_dir / f"cross_setting_mealtype_holdout_{safe_name}.csv", index=False)

    # 6) Cross-setting by baseline and activity bins if available.
    for col in ["setting_baseline_bin", "setting_activity_bin"]:
        if col in meta.columns and meta[col].notna().sum() > 0:
            labels = sorted(meta[col].dropna().unique().tolist())
            for holdout in labels:
                split = meta[["event_id", "paired_event_id", "subject_id", "device", "meal_type"]].copy()
                split["split"] = "unused"
                test_mask = meta[col] == holdout
                trainval_pairs = meta.loc[~test_mask, "paired_event_id"].drop_duplicates().to_numpy()
                if len(trainval_pairs) >= 5:
                    train_pairs, val_pairs = train_test_split(trainval_pairs, test_size=0.15, random_state=seed)
                    split.loc[(~test_mask) & (meta["paired_event_id"].isin(train_pairs)), "split"] = "train"
                    split.loc[(~test_mask) & (meta["paired_event_id"].isin(val_pairs)), "split"] = "val"
                else:
                    split.loc[~test_mask, "split"] = "train"
                split.loc[test_mask, "split"] = "test"
                split.to_csv(split_dir / f"cross_setting_{col}_holdout_{holdout}.csv", index=False)

    print(f"[OK] Split files saved to {split_dir}")
