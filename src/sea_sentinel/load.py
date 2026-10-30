"""Readers for the raw source files. These return the data as-is (strings), only fixing file-format issues."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

EMPRES_N_LEADING = 11  # Event.ID .. Longitude
EMPRES_N_TRAILING = 4  # Observation date, Report date, Humans.affected, Human.deaths
EMPRES_STATUSES = {"Confirmed", "Suspected", "Denied"}


def _find_header_row(rows: list[list[str]], first_col: str) -> int:
    for i, row in enumerate(rows):
        if row and row[0].strip() == first_col:
            return i
    raise ValueError(f"Header row starting with {first_col!r} not found")


def repair_empres_row(row: list[str], n_cols: int) -> list[str]:
    """Re-join rows whose free-text fields contain unquoted commas.

    The export writes e.g. ``National Institute for Communicable Diseases, NICD`` (Diagnosis.source)
    or ``Wild - Wild Suid (Phacochoerus, Potamochoerus )`` (Animal.type and Species) without quotes.
    The 11 leading and 4 trailing columns are reliable; in between are Diagnosis.source,
    Diagnosis.status, Animal.type and Species. The status value is anchored on its known vocabulary,
    the source is everything before it, and the two species fields split the remainder evenly.
    """
    if len(row) <= n_cols:
        return row + [""] * (n_cols - len(row))
    middle = row[EMPRES_N_LEADING:-EMPRES_N_TRAILING]
    status_pos = [i for i, v in enumerate(middle) if v.strip() in EMPRES_STATUSES]
    species = middle[status_pos[0] + 1 :] if status_pos else []
    if not status_pos or len(species) % 2:
        raise ValueError(f"Cannot repair malformed EMPRES-i row: {row[:2]}")
    s, half = status_pos[0], len(species) // 2
    return (
        row[:EMPRES_N_LEADING]
        + [",".join(middle[:s]), middle[s], ",".join(species[:half]), ",".join(species[half:])]
        + row[-EMPRES_N_TRAILING:]
    )


def load_empres(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load an EMPRES-i Plus CSV export. Returns (records, export metadata from the preamble)."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    h = _find_header_row(rows, "Event.ID")

    meta = {}
    for row in rows[:h]:
        if row and ":" in row[0]:
            key, _, value = row[0].partition(":")
            meta[key.strip()] = value.strip()

    header = rows[h]
    body = [repair_empres_row(r, len(header)) for r in rows[h + 1 :] if any(c.strip() for c in r)]
    df = pd.DataFrame(body, columns=header)
    df = df.apply(lambda s: s.str.strip()).replace("", pd.NA)
    df.attrs["n_repaired_rows"] = sum(len(r) > len(header) for r in rows[h + 1 :])
    return df, meta


def load_wahis_events(paths: list[Path]) -> pd.DataFrame:
    """Load one or more WAHIS event-list exports (semicolon separated)."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, sep=";", dtype=str, encoding="utf-8-sig", keep_default_na=False)
        df = df.apply(lambda s: s.str.strip()).replace("", pd.NA)
        df["source_file"] = p.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_faostat(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    return df.replace("", pd.NA)


def load_glw_census(path: Path) -> pd.DataFrame:
    """GLW census metadata workbook: one sheet per species (``Chicken``, ``Swine``), as text."""
    sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    return pd.concat([df.assign(sheet=name) for name, df in sheets.items()], ignore_index=True)
