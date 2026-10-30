"""Country x disease x epidemiological-week panel of reported outbreaks.

Absence of a report is not treated as zero outbreaks. For each country-disease series:

* weeks before the first or after the last reported outbreak are ``before_first_report`` /
  ``after_last_report`` and ``outbreaks`` is missing (NaN);
* weeks in between are ``active`` and an empty week counts as a reported zero;
* within the active span, runs of at least ``LONG_GAP_WEEKS`` empty weeks are flagged ``long_gap``.
  Their counts stay 0, but models may choose to treat them as surveillance gaps.

Weeks run Monday-Sunday and are labelled by their Monday (``week``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COUNTRIES, DISEASES, LONG_GAP_WEEKS


def _flag_long_gaps(zero: np.ndarray, min_len: int) -> np.ndarray:
    flag = np.zeros_like(zero, dtype=bool)
    run_start = None
    for i, z in enumerate(np.append(zero, False)):
        if z and run_start is None:
            run_start = i
        elif not z and run_start is not None:
            if i - run_start >= min_len:
                flag[run_start:i] = True
            run_start = None
    return flag


def build_weekly_panel(
    outbreaks: pd.DataFrame,
    end_week: pd.Timestamp,
    start_weeks: dict[str, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Build the full 11-country x 3-disease weekly panel.

    ``outbreaks`` are deduplicated, in-scope records with ``event_week``/``report_week`` columns.
    ``end_week`` is the last *complete* week (Monday) covered by the data extraction.
    ``start_weeks`` optionally fixes each disease's first panel week; defaults to its first record.
    """
    frames = []
    for disease in DISEASES:
        d = outbreaks[outbreaks["disease"] == disease]
        start = (start_weeks or {}).get(disease, d["event_week"].min() if len(d) else end_week)
        weeks = pd.date_range(start, end_week, freq="W-MON")
        for iso3 in COUNTRIES:
            s = d[d["iso3"] == iso3]
            by_event = s.groupby("event_week").size()
            frame = pd.DataFrame({"iso3": iso3, "disease": disease, "week": weeks})
            frame["n_reported"] = frame["week"].map(by_event).fillna(0).astype(int)
            frame["n_dated_by_report"] = (
                frame["week"].map(s[s["date_basis"] == "report"].groupby("event_week").size()).fillna(0).astype(int)
            )
            frame["n_by_report_week"] = (
                frame["week"].map(s.groupby("report_week").size()).fillna(0).astype(int)
            )

            status = np.full(len(frame), "before_first_report", dtype=object)
            if len(s):
                first, last = s["event_week"].min(), s["event_week"].max()
                active = (frame["week"] >= first) & (frame["week"] <= last)
                status[active.to_numpy()] = "active"
                status[(frame["week"] > last).to_numpy()] = "after_last_report"
                gap = _flag_long_gaps(((frame["n_reported"] == 0) & active).to_numpy(), LONG_GAP_WEEKS)
                status[gap] = "long_gap"
            frame["coverage_status"] = status
            frame["outbreaks"] = frame["n_reported"].astype(float).where(
                frame["coverage_status"].isin(["active", "long_gap"])
            )
            frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)
    panel["year"] = panel["week"].dt.isocalendar().year.astype(int)
    panel["iso_week"] = panel["week"].dt.isocalendar().week.astype(int)
    return panel
