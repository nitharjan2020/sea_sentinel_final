"""22-cell coverage audit (11 countries x 2 diseases), reporting-delay summary and forecast eligibility."""

from __future__ import annotations

import pandas as pd

from .config import COUNTRIES, DISEASES, ELIGIBILITY, TARGET_SPECIES


def _pct(num: float, den: float) -> float | None:
    return round(100 * num / den, 1) if den else None


def eligibility(n_obs_dated: int, n_total: int, active_weeks: int, nonzero_weeks: int) -> str:
    """Forecast thresholds use observation-dated outbreaks; descriptive tiers use all outbreaks."""
    meets = (
        n_obs_dated >= ELIGIBILITY["min_dated_outbreaks"]
        and active_weeks >= ELIGIBILITY["min_active_weeks"]
        and nonzero_weeks >= ELIGIBILITY["min_nonzero_weeks"]
    )
    if meets:
        return "Eligible: forecast + detection"
    if nonzero_weeks >= 10:
        return "Detection / descriptive only"
    if n_total > 0:
        return "Descriptive only (sparse)"
    return "Withhold: no dated records"


def delay_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Observation-to-report delay by country and disease, using records with plausible delays."""
    ok = records[records["delay_quality"] == "ok"]
    g = ok.groupby(["iso3", "disease"])["report_delay_days"]
    out = pd.DataFrame(
        {
            "n_with_delay": g.size(),
            "median_days": g.median(),
            "p25_days": g.quantile(0.25),
            "p75_days": g.quantile(0.75),
            "p90_days": g.quantile(0.90),
            # A high share of same-day reports can mean the observation date was set to the report date.
            "pct_zero_delay": g.apply(lambda s: 100 * (s == 0).mean()),
            "pct_within_7d": g.apply(lambda s: 100 * (s <= 7).mean()),
            "pct_within_28d": g.apply(lambda s: 100 * (s <= 28).mean()),
        }
    ).round(1)
    bad = records.groupby(["iso3", "disease"])["delay_quality"].value_counts().unstack(fill_value=0)
    for col in ("missing", "negative", "over_1y"):
        out[f"n_delay_{col}"] = bad.get(col, pd.Series(dtype=int)).reindex(out.index).fillna(0).astype(int)
    return out.reset_index()


def coverage_audit(
    standardized: pd.DataFrame,
    outbreaks: pd.DataFrame,
    panel: pd.DataFrame,
    wahis: pd.DataFrame,
) -> pd.DataFrame:
    """One row per country-disease cell, with the columns of the inventory's Coverage Matrix sheet.

    ``standardized``: all EMPRES-i rows after standardisation (in and out of scope).
    ``outbreaks``: deduplicated in-scope outbreaks. ``panel``: weekly panel. ``wahis``: WAHIS events.
    """
    rows = []
    for iso3, country in COUNTRIES.items():
        for disease in DISEASES:
            raw = standardized[(standardized["iso3"] == iso3) & (standardized["disease"] == disease)]
            inc = raw[raw["include"]]
            ob = outbreaks[(outbreaks["iso3"] == iso3) & (outbreaks["disease"] == disease)]
            pn = panel[(panel["iso3"] == iso3) & (panel["disease"] == disease)]
            wh = wahis[(wahis["iso3"] == iso3) & (wahis["disease"] == disease)]

            active = pn["coverage_status"].isin(["active", "long_gap"])
            active_weeks = int(active.sum())
            nonzero_weeks = int((pn["n_reported"] > 0).sum())
            long_gap_weeks = int((pn["coverage_status"] == "long_gap").sum())
            n_obs_dated = int((ob["date_basis"] == "observation").sum())

            excluded = raw.loc[~raw["include"], "exclude_reason"].value_counts()
            notes = []
            if len(wh):
                notes.append(
                    f"WAHIS: {len(wh)} event(s) {wh['event_start_date'].min():%Y}-{wh['event_start_date'].max():%Y}"
                )
            if len(excluded):
                notes.append("excluded: " + "; ".join(f"{k} {v}" for k, v in excluded.items()))
            if len(ob):
                media = (ob["reporting_source"] == "Media").mean()
                if media > 0.1:
                    notes.append(f"{media:.0%} media-sourced")

            src = ["SRC003"] if len(raw) else []
            if len(wh):
                src.append("SRC001")

            rows.append(
                {
                    "ISO3": iso3,
                    "Country": country,
                    "Disease": disease,
                    "Target species": TARGET_SPECIES[disease],
                    "Assessment status": "Assessed" if len(raw) or len(wh) else "Assessed: no records",
                    "First date": ob["event_date"].min() if len(ob) else pd.NaT,
                    "Last date": ob["event_date"].max() if len(ob) else pd.NaT,
                    "Event records": len(raw),
                    "In-scope records": len(inc),
                    "Distinct outbreaks": len(ob),
                    "Usable dates (%)": _pct(n_obs_dated, len(ob)),
                    "Report dates (%)": _pct(int(ob["report_date"].notna().sum()), len(ob)),
                    "Time resolution": "Daily (observation date; report date fallback)" if len(ob) else "",
                    "Active weeks": active_weeks,
                    "Weeks with outbreaks": nonzero_weeks,
                    "Long-gap weeks": long_gap_weeks,
                    "Zero vs missing": (
                        f"Zeros only within first-last report window ({active_weeks} wk); "
                        "outside window = missing"
                        if active_weeks
                        else "No reports: all weeks missing, not zero"
                    ),
                    "Forecast eligibility": eligibility(n_obs_dated, len(ob), active_weeks, nonzero_weeks),
                    "WAHIS events": len(wh),
                    "Source ID(s)": "; ".join(src),
                    "Notes": " | ".join(notes),
                }
            )
    return pd.DataFrame(rows)
