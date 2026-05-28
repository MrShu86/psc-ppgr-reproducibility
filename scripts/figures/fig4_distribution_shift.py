from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "final"
OUT_BASE = OUT_DIR / "Fig4_distribution_shift"

DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "events_metadata.csv",
    ROOT / "outputs" / "events_metadata.csv",
]

RNG = np.random.default_rng(42)


# Palette copied from Fig. 2. Do not introduce new colors here; keeping
# Fig. 4 visually consistent with the manuscript figure set is intentional.
COLORS = {
    "data": "#D9EAF7",
    "event": "#DFF1E4",
    "shift": "#FCE5CD",
    "model": "#E8E1F5",
    "psc": "#D8F0F2",
    "output": "#ECEFF3",
    "blue": "#2C7FB8",
    "text": "#1F2D3D",
    "muted": "#566573",
    "border": "#34495E",
    "grid": "#DDE4EA",
    "white": "#FFFFFF",
    "green": "#2A9D8F",
    "orange": "#E99A3A",
    "purple": "#7A6BB7",
    "red": "#C75D5D",
    "beige": "#F8E8C8",
    "meal_blue": "#3B73B9",
}

MEAL_COLORS = {
    "breakfast": "#3B73B9",
    "lunch": "#2A9D8F",
    "dinner": "#E99A3A",
    "snack": "#7A6BB7",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
        }
    )


def clean_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.8, length=3, color=COLORS["border"])
    ax.spines["left"].set_color(COLORS["border"])
    ax.spines["bottom"].set_color(COLORS["border"])
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.55, alpha=0.75)


def panel_title(ax, letter, title):
    ax.text(
        -0.105,
        1.055,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        -0.005,
        1.065,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.4,
        fontweight="bold",
        color=COLORS["text"],
    )


def first_existing_column(df, candidates):
    lower_to_original = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]
    return None


def find_input_csv():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    return None


def build_template_data(n=3370):
    # Template only. Replace with a real CSV containing:
    # subject_id, device, meal_type, and iauc_2h or target_iauc_2h.
    subjects = [f"S{i:02d}" for i in range(1, 46)]
    devices = RNG.choice(["dexcom", "libre"], size=n, p=[0.50, 0.50])
    meals = RNG.choice(["breakfast", "lunch", "dinner", "snack"], size=n, p=[0.25, 0.26, 0.29, 0.20])
    subj = RNG.choice(subjects, size=n)
    subject_effect = {s: RNG.normal(0, 650) for s in subjects}
    meal_effect = {"breakfast": 1300, "lunch": 450, "dinner": 620, "snack": -350}
    device_effect = {"dexcom": 220, "libre": 0}
    base = RNG.gamma(shape=1.55, scale=1600, size=n)
    iauc = np.clip(
        base
        + np.array([subject_effect[s] for s in subj])
        + np.array([meal_effect[m] for m in meals])
        + np.array([device_effect[d] for d in devices]),
        0,
        None,
    )
    return pd.DataFrame({"subject_id": subj, "device": devices, "meal_type": meals, "iauc_2h": iauc}), True, "template"


def load_events():
    path = find_input_csv()
    if path is None:
        return build_template_data()

    df = pd.read_csv(path)
    colmap = {
        "subject": first_existing_column(df, ["subject_id", "subject", "participant_id"]),
        "device": first_existing_column(df, ["device", "device_name", "sensor", "device_col"]),
        "meal": first_existing_column(df, ["meal_type", "setting_meal_type", "meal_type_raw"]),
        "target": first_existing_column(df, ["iauc_2h", "target_iauc_2h", "iAUC2h", "iauc120"]),
        "baseline": first_existing_column(df, ["baseline_glucose", "baseline", "premeal_glucose"]),
    }

    target = colmap["target"] or colmap["baseline"]
    required = [colmap["subject"], colmap["device"], colmap["meal"], target]
    if any(col is None for col in required):
        return build_template_data()

    out = pd.DataFrame(
        {
            "subject_id": df[colmap["subject"]].astype(str),
            "device": df[colmap["device"]].astype(str).str.lower(),
            "meal_type": df[colmap["meal"]].astype(str).str.lower(),
            "value": pd.to_numeric(df[target], errors="coerce"),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
    out = out[out["value"] >= 0].copy()
    out["device"] = out["device"].replace({"dexcom gl": "dexcom", "libre gl": "libre"})
    out["meal_type"] = out["meal_type"].str.strip()
    out = out[out["device"].isin(["dexcom", "libre"])]
    out = out[out["meal_type"].isin(["breakfast", "lunch", "dinner", "snack"])]
    out.attrs["target_label"] = "iAUC2h (mg/dL*min)" if target == colmap["target"] else "Baseline glucose (mg/dL)"
    return out, False, str(path)


def make_boxplot(ax, groups, labels, colors, y_label=None):
    bp = ax.boxplot(
        groups,
        positions=np.arange(1, len(groups) + 1),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=COLORS["border"], linewidth=1.25),
        whiskerprops=dict(color=COLORS["border"], linewidth=0.9),
        capprops=dict(color=COLORS["border"], linewidth=0.9),
        boxprops=dict(color=COLORS["border"], linewidth=0.95),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    for idx, values in enumerate(groups, start=1):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            continue
        if len(values) > 180:
            values = RNG.choice(values, size=180, replace=False)
        jitter = RNG.normal(0, 0.040, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=5,
            color=COLORS["border"],
            alpha=0.12,
            linewidths=0,
            rasterized=True,
        )

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    if y_label:
        ax.set_ylabel(y_label)
    clean_axis(ax, "y")


def draw_panel_a(ax, df, target_label, template):
    panel_title(ax, "A", "Device shift")
    order = ["dexcom", "libre"]
    labels = ["Dexcom", "Libre"]
    groups = [df.loc[df["device"].eq(dev), "value"].to_numpy() for dev in order]
    colors = [COLORS["meal_blue"], COLORS["green"]]
    make_boxplot(ax, groups, labels, colors, y_label=target_label)
    ax.set_xlabel("CGM device")
    if template:
        ax.text(0.02, 0.96, "TEMPLATE", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=COLORS["muted"])


def draw_panel_b(ax, df, target_label, template):
    panel_title(ax, "B", "Subject heterogeneity")
    stats = (
        df.groupby("subject_id")["value"]
        .agg(q25=lambda x: np.percentile(x, 25), median="median", q75=lambda x: np.percentile(x, 75), n="size")
        .query("n >= 2")
        .sort_values("median")
        .reset_index()
    )
    x = np.arange(1, len(stats) + 1)
    lower = stats["median"].to_numpy() - stats["q25"].to_numpy()
    upper = stats["q75"].to_numpy() - stats["median"].to_numpy()
    ax.errorbar(
        x,
        stats["median"],
        yerr=[lower, upper],
        fmt="o",
        ms=3.0,
        color=COLORS["blue"],
        ecolor=COLORS["data"],
        elinewidth=1.2,
        capsize=0,
        alpha=0.95,
    )
    ax.set_xlabel("Subjects sorted by median response")
    ax.set_ylabel(target_label)
    if len(stats) > 0:
        ticks = np.unique(np.round(np.linspace(1, len(stats), min(5, len(stats)))).astype(int))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
    clean_axis(ax, "y")
    if template:
        ax.text(0.02, 0.96, "TEMPLATE", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=COLORS["muted"])


def draw_panel_c(ax, df, target_label, template):
    panel_title(ax, "C", "Meal-setting shift")
    order = ["breakfast", "lunch", "dinner", "snack"]
    labels = ["Breakfast", "Lunch", "Dinner", "Snack"]
    groups = [df.loc[df["meal_type"].eq(meal), "value"].to_numpy() for meal in order]
    colors = [MEAL_COLORS[m] for m in order]
    make_boxplot(ax, groups, labels, colors, y_label=target_label)
    ax.set_xlabel("Meal setting")
    ax.tick_params(axis="x", rotation=18)
    if template:
        ax.text(0.02, 0.96, "TEMPLATE", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=COLORS["muted"])


def main():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, template, source = load_events()
    if "value" not in df.columns:
        df = df.rename(columns={"iauc_2h": "value"})
    target_label = df.attrs.get("target_label", "iAUC2h (mg/dL*min)")

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), constrained_layout=False)
    draw_panel_a(axes[0], df, target_label, template)
    draw_panel_b(axes[1], df, target_label, template)
    draw_panel_c(axes[2], df, target_label, template)

    finite = df["value"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    raw_y_max = np.max(finite) * 1.05 if len(finite) else 1.0

    # Panels A and C show raw event distributions, so they keep the full
    # event-level range. Panel B summarizes subject-level medians and IQRs;
    # using the raw-event maximum would compress the subject trend visually.
    subject_q75 = df.groupby("subject_id")["value"].quantile(0.75).to_numpy(dtype=float)
    subject_y_max = np.nanmax(subject_q75) * 1.18 if len(subject_q75) else raw_y_max
    axes[0].set_ylim(0, raw_y_max)
    axes[1].set_ylim(0, min(raw_y_max, subject_y_max))
    axes[2].set_ylim(0, raw_y_max)
    for ax in axes:
        ax.yaxis.label.set_color(COLORS["text"])
        ax.xaxis.label.set_color(COLORS["text"])

    fig.subplots_adjust(left=0.070, right=0.992, top=0.855, bottom=0.220, wspace=0.36)

    for ext in ("png", "svg", "pdf"):
        out = OUT_BASE.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.05)
        else:
            fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"Saved: {OUT_BASE.with_suffix('.png')}")
    print(f"Saved: {OUT_BASE.with_suffix('.pdf')}")
    print(f"Saved: {OUT_BASE.with_suffix('.svg')}")
    print(f"Data source: {source}")


if __name__ == "__main__":
    main()
