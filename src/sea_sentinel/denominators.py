"""At-risk livestock populations from FAOSTAT annual stocks, attached to the weekly panel as offsets.

* HPAI: chickens + ducks + geese + turkeys; ASF: pigs (``config.POPULATION_ITEMS``).
  Items FAOSTAT does not report for a country add nothing.
* Annual stocks are anchored at 1 July and log-linearly interpolated to each week, so the offset
  changes smoothly instead of jumping on 1 January.
* Look-ahead: FAOSTAT publishes with a lag of 1-2 years. With ``lag_years = k`` the stock of year Y
  is anchored at 1 July of Y + k, so a week only sees stocks that would already have been published.
  The pipeline uses ``config.POPULATION_LAG_YEARS`` (2).
* Weeks beyond the first/last anchor carry the nearest value and are flagged
  ``population_extrapolated``.
* A country-disease with no reported stock (e.g. Singapore pigs: "data cannot exist") gets a missing
  population, not zero, so a log offset is never -inf.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import POPULATION_ITEMS


def annual_population(stocks: pd.DataFrame) -> pd.DataFrame:
    """iso3 x disease x year at-risk population (head) with the share of official FAOSTAT values."""
    frames = []
    for disease, items in POPULATION_ITEMS.items():
        s = stocks[stocks["item"].isin(items) & stocks["head_count"].notna()]
        if s.empty:
            continue
        g = s.groupby(["iso3", "year"])
        frames.append(
            pd.DataFrame(
                {
                    "population_head": g["head_count"].sum(),
                    "n_items": g["item"].nunique(),
                    "pct_official": g.apply(
                        lambda x: 100 * x.loc[x["flag"] == "A", "head_count"].sum() / x["head_count"].sum(),
                        include_groups=False,
                    ),
                }
            )
            .reset_index()
            .assign(disease=disease)
        )
    out = pd.concat(frames, ignore_index=True)
    return out[out["population_head"] > 0][["iso3", "disease", "year", "population_head", "n_items", "pct_official"]]


def add_population(panel: pd.DataFrame, annual: pd.DataFrame, lag_years: int = 0) -> pd.DataFrame:
    """Add ``population_head``, ``log_population`` and ``population_extrapolated`` to the weekly panel.

    With ``lag_years = k`` the stock of year Y is anchored at 1 July of Y + k.
    """
    panel = panel.copy()
    panel["population_head"] = np.nan
    panel["population_extrapolated"] = False

    for (iso3, disease), a in annual.groupby(["iso3", "disease"]):
        rows = (panel["iso3"] == iso3) & (panel["disease"] == disease)
        if not rows.any():
            continue
        a = a.sort_values("year")
        anchor = pd.to_datetime((a["year"] + lag_years).astype(str) + "-07-01").to_numpy().astype("datetime64[D]").astype(float)
        week = panel.loc[rows, "week"].to_numpy().astype("datetime64[D]").astype(float)
        # np.interp holds the end values flat outside the anchor range.
        logpop = np.interp(week, anchor, np.log(a["population_head"].to_numpy(float)))
        panel.loc[rows, "population_head"] = np.exp(logpop).round()
        panel.loc[rows, "population_extrapolated"] = (week < anchor[0]) | (week > anchor[-1])

    panel["log_population"] = np.log(panel["population_head"])
    return panel


def census_check(census: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    """GLW national census totals against FAOSTAT stocks of the same item and year.

    ``ratio`` = census / FAOSTAT. The detection model only uses the offset's change over time, so a
    constant ratio would not matter; a large one shows how uncertain the levels are.
    """
    f = stocks.rename(columns={"year": "census_year", "head_count": "faostat_head", "flag": "faostat_flag"})
    out = census.merge(f[["iso3", "item", "census_year", "faostat_head", "faostat_flag"]],
                       on=["iso3", "item", "census_year"], how="left")
    out["ratio"] = out["census_head"] / out["faostat_head"]
    return out


def population_summary(annual: pd.DataFrame) -> pd.DataFrame:
    """Per country-disease: years covered, latest population and share of official values."""
    g = annual.sort_values("year").groupby(["iso3", "disease"])
    return pd.DataFrame(
        {
            "first_year": g["year"].min(),
            "last_year": g["year"].max(),
            "latest_population_head": g["population_head"].last(),
            "items_used": g["n_items"].max(),
            "pct_official_mean": g["pct_official"].mean().round(1),
        }
    ).reset_index()
