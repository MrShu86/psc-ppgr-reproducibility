from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .io_utils import ensure_dir


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def profile_dataset(config: dict) -> None:
    out = Path(config["output_dir"])
    fig_dir = ensure_dir(out / "profiling")
    meta_path = out / "events_metadata.csv"
    seq_path = out / "events_sequences.npz"

    if not meta_path.exists() or not seq_path.exists():
        raise FileNotFoundError("Run 01_build_meal_events.py first.")

    meta = pd.read_csv(meta_path)
    arr = np.load(seq_path)
    seq = arr["sequences"]
    rel_grid = arr["rel_grid"]

    main_target = config.get("main_target", "iauc_2h")

    # 1) Device target distribution.
    plt.figure(figsize=(7, 4))
    for device, sub in meta.groupby("device"):
        vals = pd.to_numeric(sub[main_target], errors="coerce").dropna()
        plt.hist(vals, bins=30, alpha=0.5, label=device)
    plt.xlabel(main_target)
    plt.ylabel("Count")
    plt.title("Target distribution by device")
    plt.legend()
    _savefig(fig_dir / "device_target_distribution.png")

    # 2) Bland-Altman for paired Libre/Dexcom targets.
    paired = meta.pivot_table(index="paired_event_id", columns="device", values=main_target, aggfunc="mean")
    if {"libre", "dexcom"}.issubset(set(paired.columns)):
        mean_vals = paired[["libre", "dexcom"]].mean(axis=1)
        diff_vals = paired["libre"] - paired["dexcom"]
        plt.figure(figsize=(6, 4))
        plt.scatter(mean_vals, diff_vals, s=12, alpha=0.7)
        plt.axhline(diff_vals.mean(), linestyle="--")
        plt.axhline(diff_vals.mean() + 1.96 * diff_vals.std(), linestyle=":")
        plt.axhline(diff_vals.mean() - 1.96 * diff_vals.std(), linestyle=":")
        plt.xlabel(f"Mean {main_target}")
        plt.ylabel(f"Libre - Dexcom {main_target}")
        plt.title("Bland-Altman style device difference")
        _savefig(fig_dir / "device_bland_altman.png")

    # 3) Subject heterogeneity.
    subject_vals = meta.groupby("subject_id")[main_target].mean().sort_values()
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(subject_vals)), subject_vals.values)
    plt.xticks(np.arange(len(subject_vals)), subject_vals.index, rotation=90, fontsize=7)
    plt.ylabel(f"Mean {main_target}")
    plt.title("Subject-level heterogeneity")
    _savefig(fig_dir / "subject_heterogeneity.png")

    # 4) Meal type distribution.
    if "meal_type" in meta.columns:
        groups = []
        labels = []
        for label, sub in meta.groupby("meal_type"):
            vals = pd.to_numeric(sub[main_target], errors="coerce").dropna()
            if len(vals) > 1:
                groups.append(vals.values)
                labels.append(str(label))
        if groups:
            plt.figure(figsize=(7, 4))
            plt.boxplot(groups, labels=labels, showfliers=False)
            plt.ylabel(main_target)
            plt.title("PPGR target by meal type")
            plt.xticks(rotation=30)
            _savefig(fig_dir / "meal_type_targets.png")

    # 5) Nutrition-target correlation heatmap.
    nutrition_cols = [c for c in config.get("nutrition_columns", []) if c in meta.columns]
    targets = [c for c in ["peak_rise", "iauc_2h", "iauc_3h", "time_to_peak", "recovery_slope", "hyper_duration_140"] if c in meta.columns]
    cols = nutrition_cols + targets
    if len(cols) >= 2:
        corr = meta[cols].apply(pd.to_numeric, errors="coerce").corr()
        plt.figure(figsize=(max(7, len(cols) * 0.55), max(5, len(cols) * 0.45)))
        im = plt.imshow(corr, aspect="auto")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(np.arange(len(cols)), cols, rotation=90)
        plt.yticks(np.arange(len(cols)), cols)
        plt.title("Nutrition-response correlation")
        _savefig(fig_dir / "nutrition_target_correlation.png")

    # 6) Average response curves by device.
    plt.figure(figsize=(7, 4))
    for device, sub in meta.groupby("device"):
        idx = sub.index.to_numpy()
        mean_curve = np.nanmean(seq[idx], axis=0)
        plt.plot(rel_grid, mean_curve, label=device)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Minutes from meal")
    plt.ylabel("Glucose")
    plt.title("Mean meal-centered CGM curve by device")
    plt.legend()
    _savefig(fig_dir / "device_mean_curves.png")

    # Summary CSV.
    summary = {
        "n_device_level_events": len(meta),
        "n_paired_meal_events": meta["paired_event_id"].nunique(),
        "n_subjects": meta["subject_id"].nunique(),
        "devices": "; ".join([f"{k}:{v}" for k, v in meta["device"].value_counts().to_dict().items()]),
        "main_target": main_target,
        "main_target_mean": float(pd.to_numeric(meta[main_target], errors="coerce").mean()),
        "main_target_std": float(pd.to_numeric(meta[main_target], errors="coerce").std()),
    }
    pd.DataFrame([summary]).to_csv(fig_dir / "dataset_summary.csv", index=False)

    print(f"[OK] Profiling figures saved to {fig_dir}")
