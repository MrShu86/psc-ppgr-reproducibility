from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml
import pandas as pd
import numpy as np


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_subject_id(name: str) -> str:
    """
    Converts folder/file names like CGMacros-1, CGMacros-001, CGMacros_001 to CGMacros-001.
    """
    m = re.search(r"CGMacros[-_ ]?(\d+)", name, flags=re.IGNORECASE)
    if m:
        return f"CGMacros-{int(m.group(1)):03d}"
    m = re.search(r"(\d+)", name)
    if m:
        return f"CGMacros-{int(m.group(1)):03d}"
    return name


def find_participant_csvs(data_root: str | Path) -> List[Tuple[str, Path]]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")

    candidates: List[Path] = []
    for pattern in ["CGMacros-*/CGMacros-*.csv", "CGMacros_*/CGMacros_*.csv", "**/CGMacros-*.csv", "**/CGMacros_*.csv"]:
        candidates.extend(root.glob(pattern))

    # Remove data dictionary and supplementary files.
    filtered = []
    for p in candidates:
        low = p.name.lower()
        if "datadictionary" in low or low in {"bio.csv", "microbes.csv"}:
            continue
        if "cgmacros" in low and p.is_file():
            filtered.append(p)

    unique = sorted(set(filtered))
    out = []
    for p in unique:
        sid = normalize_subject_id(p.stem)
        out.append((sid, p))
    return out


def find_supplement_file(data_root: str | Path, keywords: List[str]) -> Optional[Path]:
    root = Path(data_root)
    all_csvs = list(root.glob("*.csv")) + list(root.glob("**/*.csv"))
    for p in all_csvs:
        low = p.name.lower()
        if all(k.lower() in low for k in keywords):
            return p
    return None


def load_bio_table(data_root: str | Path) -> pd.DataFrame:
    """
    Loads bio.csv if available. Returns empty DataFrame otherwise.
    The subject column is normalized to subject_id.
    """
    root = Path(data_root)
    possible = []
    for name in ["bio.csv", "Bio.csv", "BIO.csv"]:
        possible.extend(root.glob(name))
        possible.extend(root.glob(f"**/{name}"))

    if not possible:
        return pd.DataFrame()

    p = sorted(possible)[0]
    bio = pd.read_csv(p)

    # Try to find subject column.
    subject_col = None
    for c in bio.columns:
        if c.lower() in {"subject", "subject id", "participant", "participant id", "id"}:
            subject_col = c
            break
        if "cgmacros" in c.lower():
            subject_col = c
            break

    if subject_col is None:
        # If there are 45 rows and no ID column, assign sequential IDs.
        bio = bio.copy()
        bio["subject_id"] = [f"CGMacros-{i+1:03d}" for i in range(len(bio))]
    else:
        bio = bio.copy()
        bio["subject_id"] = bio[subject_col].astype(str).map(normalize_subject_id)

    return bio


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps original names but strips whitespace.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")
