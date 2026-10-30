"""Copy original downloads into data/raw/ under canonical names and record their provenance.

Raw files are never modified after ingestion; every later stage reads from data/raw/.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import RAW_DIR, RAW_SOURCES, SOURCE_DIR


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_path(raw_name: str) -> Path:
    return RAW_DIR / raw_name


def ingest(source_dir: Path = SOURCE_DIR, force: bool = False) -> pd.DataFrame:
    """Copy each registered source file and write data/raw/MANIFEST.csv. Returns the manifest."""
    records = []
    for src in RAW_SOURCES:
        origin = source_dir / src.original_name
        dest = raw_path(src.raw_name)
        if not dest.exists() or force:
            if not origin.exists():
                raise FileNotFoundError(
                    f"{origin} not found. Set SEA_SENTINEL_SOURCE_DIR to the folder with the downloads."
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, dest)
        records.append(
            {
                **asdict(src),
                "bytes": dest.stat().st_size,
                "sha256": sha256(dest),
            }
        )
    manifest = pd.DataFrame(records)
    manifest.to_csv(RAW_DIR / "MANIFEST.csv", index=False)
    return manifest
