"""Collapse repeated reports of the same outbreak into one record.

Two passes over in-scope EMPRES-i records:

1. **Exact repeats** - same country, disease, admin1, locality, coordinates (4 dp, ~10 m), event date
   and reporting source. This catches re-uploads and the per-species rows of a single outbreak (e.g.
   one village record for chickens and another for ducks on the same day). All animal types are
   kept in ``animal_types``.
2. **Cross-source repeats** - the same outbreak reported through different channels (e.g. national
   authorities and WOAH): same disease, great-circle distance <= ``max_km`` and event dates within
   ``window_days``. Candidate pairs are merged closest-first, and a group never holds two records from
   the same source, because records from one channel at nearby places are distinct outbreaks (WOAH
   rows, for instance, carry unique outbreak IDs).

The thresholds come from ``config`` and are justified by ``dedup_review``. Within a group the kept
record is the most authoritative (``SOURCE_RANK``), then one with an observation date, then the
earliest report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEDUP_MAX_KM, DEDUP_WINDOW_DAYS

EARTH_RADIUS_KM = 6371.0

SOURCE_RANK = {
    "National authorities": 0,
    "WOAH (former OIE)": 1,
    "International reference laboratory": 2,
    "FAO Field Officer": 3,
    "FAO officer": 3,
    "WHO": 4,
    "Other": 5,
    "Media": 6,
}


def _priority(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        _src_rank=df["reporting_source"].map(SOURCE_RANK).fillna(9),
        _has_obs=df["observation_date"].isna().astype(int),
        _rep=df["report_date"].fillna(pd.Timestamp.max),
    ).sort_values(["_src_rank", "_has_obs", "_rep", "record_id"])


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def candidate_pairs(df: pd.DataFrame, max_km: float, window_days: int) -> pd.DataFrame:
    """All record pairs of the same disease within ``window_days`` and ``max_km`` of each other.

    Returns columns ``i``, ``j`` (index labels of ``df``), ``lag_days``, ``km`` and ``cross_source``.
    """
    out = []
    for _, g in df.groupby("disease", sort=False):
        g = g.sort_values("event_date")
        day = g["event_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
        lat, lon = g["latitude"].to_numpy(float), g["longitude"].to_numpy(float)
        src = g["reporting_source"].fillna("").to_numpy()
        idx = g.index.to_numpy()
        stop = np.searchsorted(day, day + window_days, side="right")
        for a in range(len(g)):
            b = np.arange(a + 1, stop[a])
            if not len(b):
                continue
            km = haversine_km(lat[a], lon[a], lat[b], lon[b])
            near = km <= max_km
            if near.any():
                b = b[near]
                out.append(
                    pd.DataFrame(
                        {
                            "i": idx[a],
                            "j": idx[b],
                            "lag_days": day[b] - day[a],
                            "km": km[near],
                            "cross_source": src[b] != src[a],
                        }
                    )
                )
    if out:
        return pd.concat(out, ignore_index=True)
    return pd.DataFrame(
        {
            "i": pd.Series([], dtype=df.index.dtype),
            "j": pd.Series([], dtype=df.index.dtype),
            "lag_days": pd.Series([], dtype="int64"),
            "km": pd.Series([], dtype=float),
            "cross_source": pd.Series([], dtype=bool),
        }
    )


def _exact_groups(df: pd.DataFrame) -> pd.Series:
    key = pd.DataFrame(
        {
            "iso3": df["iso3"],
            "disease": df["disease"],
            "admin1": df["admin1"].fillna("").str.lower(),
            "locality": df["locality"].fillna("").str.lower(),
            "lat": df["latitude"].round(4),
            "lon": df["longitude"].round(4),
            "date": df["event_date"],
            "source": df["reporting_source"].fillna(""),
        }
    )
    return key.groupby(list(key.columns), dropna=False, sort=False).ngroup()


def _cross_source_groups(df: pd.DataFrame, max_km: float, window_days: int) -> pd.Series:
    """Greedy closest-first linkage of cross-source pairs, never joining two records of one source."""
    pairs = candidate_pairs(df, max_km, window_days)
    pairs = pairs.loc[pairs["cross_source"].astype(bool)].sort_values(["km", "lag_days"])

    parent = {i: i for i in df.index}
    sources = {i: {s} for i, s in df["reporting_source"].fillna("").items()}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(pairs["i"], pairs["j"]):
        ri, rj = root(i), root(j)
        if ri != rj and not (sources[ri] & sources[rj]):
            parent[rj] = ri
            sources[ri] |= sources.pop(rj)
    return pd.Series({i: root(i) for i in df.index}).reindex(df.index)


def _merge(df: pd.DataFrame, group_col: str, carry: dict[str, str]) -> pd.DataFrame:
    """Keep the top-priority row per group; ``carry`` maps a source column to a '|'-joined list column."""
    df = _priority(df)
    grouped = df.groupby(group_col, sort=False)
    kept = grouped.head(1).copy()
    for src_col, list_col in carry.items():
        joined = grouped[src_col].agg(
            lambda s: "|".join(dict.fromkeys(x.strip() for v in s.dropna() for x in str(v).split("|")))
        )
        kept[list_col] = kept[group_col].map(joined)
    return kept


def merge_exact(records: pd.DataFrame) -> pd.DataFrame:
    """Pass 1 only: collapse exact repeats (helper columns ``_ids`` and ``_animals`` are added)."""
    df = records.copy()
    df["_g1"] = _exact_groups(df)
    return _merge(df, "_g1", {"record_id": "_ids", "animal_type": "_animals"})


def deduplicate(
    records: pd.DataFrame,
    max_km: float = DEDUP_MAX_KM,
    window_days: int = DEDUP_WINDOW_DAYS,
) -> pd.DataFrame:
    """Return one row per distinct outbreak with ``n_merged``, ``merged_record_ids``, ``animal_types``
    and ``reporting_sources`` columns.

    Expects standardized, in-scope records (``include == True``) with an ``event_date``.
    """
    kept = merge_exact(records)
    kept["_g2"] = _cross_source_groups(kept, max_km, window_days)
    out = _merge(
        kept,
        "_g2",
        {"_ids": "merged_record_ids", "_animals": "animal_types", "reporting_source": "reporting_sources"},
    )
    out["n_merged"] = out["merged_record_ids"].str.count(r"\|") + 1

    helper = [c for c in out.columns if c.startswith("_")]
    return out.drop(columns=helper).sort_values(["iso3", "disease", "event_date"]).reset_index(drop=True)
