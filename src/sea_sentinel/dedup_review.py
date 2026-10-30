"""Evidence for the cross-source deduplication thresholds (``DEDUP_MAX_KM``, ``DEDUP_WINDOW_DAYS``).

Idea: pairs of nearby records from the *same* reporting channel are distinct outbreaks, so they show
how outbreaks cluster in space and time. Pairs from *different* channels contain the same clustering
plus true duplicates. The cross/same-source pair ratio is therefore flat where there are no duplicates
and elevated where duplicates sit; the thresholds should cover the elevated region and no more.

Run ``python -m sea_sentinel.dedup_review`` after the main pipeline (it reads
``data/interim/empres_standardized.csv``).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import DEDUP_MAX_KM, DEDUP_WINDOW_DAYS, DISEASES, INTERIM_DIR, TABLE_DIR
from .dedup import candidate_pairs, deduplicate, merge_exact

log = logging.getLogger("sea_sentinel")

LAG_BINS = [-1, 0, 1, 3, 7, 14, 21, 28, 42, 60]
KM_BINS = [0, 0.1, 0.5, 1, 2, 3, 5, 10]
GRID_KM = [0.5, 1, 2, 5]
GRID_DAYS = [0, 1, 3, 7, 14]


def _ratio_profile(pairs: pd.DataFrame, col: str, bins: list[float]) -> pd.DataFrame:
    cut = pd.cut(pairs[col], bins, include_lowest=True)
    t = pairs.groupby([cut, "cross_source"], observed=False).size().unstack(fill_value=0)
    t = t.reindex(columns=[False, True], fill_value=0)
    out = pd.DataFrame(
        {
            "lower": bins[:-1],
            "upper": bins[1:],
            "same_source_pairs": t[False].to_numpy(),
            "cross_source_pairs": t[True].to_numpy(),
        }
    )
    out["cross_to_same_ratio"] = (out["cross_source_pairs"] / out["same_source_pairs"].replace(0, np.nan)).round(3)
    return out


def review(standardized: pd.DataFrame) -> dict[str, pd.DataFrame]:
    inc = standardized[standardized["include"]]
    pass1 = merge_exact(inc)

    pairs = candidate_pairs(pass1, max_km=max(KM_BINS), window_days=max(LAG_BINS))
    lag = _ratio_profile(pairs[pairs["km"] <= DEDUP_MAX_KM], "lag_days", LAG_BINS)
    dist = _ratio_profile(pairs[pairs["lag_days"] <= DEDUP_WINDOW_DAYS], "km", KM_BINS)

    rows = []
    for km in GRID_KM:
        for days in GRID_DAYS:
            d = deduplicate(inc, max_km=km, window_days=days)
            row = {"max_km": km, "window_days": days, "total": len(d)}
            row.update(d["disease"].value_counts().reindex(list(DISEASES), fill_value=0).to_dict())
            rows.append(row)
    grid = pd.DataFrame(rows)
    grid["chosen"] = (grid["max_km"] == DEDUP_MAX_KM) & (grid["window_days"] == DEDUP_WINDOW_DAYS)
    ref = grid.loc[grid["chosen"], "total"].iloc[0]
    grid["pct_vs_chosen"] = (100 * (grid["total"] - ref) / ref).round(2)

    summary = pd.DataFrame(
        {
            "stage": ["in-scope records", "after exact repeats", "after cross-source repeats"],
            "records": [len(inc), len(pass1), ref],
        }
    )
    return {"lag": lag, "distance": dist, "grid": grid, "summary": summary}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    std = pd.read_csv(
        INTERIM_DIR / "empres_standardized.csv",
        parse_dates=["observation_date", "report_date", "event_date"],
        dtype={"record_id": str},
    )
    res = review(std)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    res["lag"].to_csv(TABLE_DIR / "dedup_lag_profile.csv", index=False)
    res["distance"].to_csv(TABLE_DIR / "dedup_distance_profile.csv", index=False)
    res["grid"].to_csv(TABLE_DIR / "dedup_sensitivity.csv", index=False)
    res["summary"].to_csv(TABLE_DIR / "dedup_summary.csv", index=False)

    from .plots import dedup_review_figure

    dedup_review_figure(res["lag"], res["distance"], DEDUP_MAX_KM, DEDUP_WINDOW_DAYS)
    for name in ("summary", "lag", "distance", "grid"):
        log.info("\n%s\n%s", name, res[name].to_string(index=False))


if __name__ == "__main__":
    main()
