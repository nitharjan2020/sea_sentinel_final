"""Covariates from the open climate and socio-economic downloads (``download.py``).

**Climate (NOAA CPC, SRC011).** The raw snapshot holds daily precipitation and daily maximum /
minimum temperature averaged over each country's land cells. Per country and panel week
(Monday-Sunday):

* ``tmean`` = mean over the week's days of (tmax + tmin) / 2, in degC;
* ``precip`` = weekly total in mm (the daily mean x 7 when days are missing).

Anomalies remove the normal seasonal cycle: for each country and week of the year, the mean and SD
over the reference years ``CLIMATOLOGY_YEARS`` (all before the forecasting held-out period, so the
held-out evaluation uses no later information). Temperature anomalies are z-scores of ``tmean``;
rainfall anomalies are z-scores of log(1 + precip). The forecasting predictors at week T are the
mean anomalies of weeks T-3..T (``temp_anom_4w``, ``precip_anom_4w``), all known at T.

**Socio-economic (World Bank WDI, SRC007).** One row per country and year with population by sex,
male and female population density (population / land area), population density, GDP, GDP per
capita and trade. These are context for the coverage audit and future risk models: annual
national values barely change within a forecast's training window, so they are not used as weekly
forecast predictors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .standardize import week_start

CLIMATOLOGY_YEARS = (2003, 2021)
ANOMALY_WINDOW = 4
CLIMATE_PREDICTORS = ("temp_anom_4w", "precip_anom_4w")


def climate_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Country x week climate with anomalies and the 4-week forecast predictors."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["week"] = week_start(d["date"])
    d["tday"] = (d["tmax"] + d["tmin"]) / 2
    g = d.groupby(["iso3", "week"])
    w = pd.DataFrame({"tmean": g["tday"].mean(), "precip": 7 * g["precip"].mean(),
                      "days": g["date"].nunique()}).reset_index()
    w = w[w["days"] >= 4].drop(columns="days")  # a week with fewer days (start or end of the record) is left out
    w["woy"] = np.minimum(w["week"].dt.isocalendar().week.astype(int), 52)  # week 53 joins week 52
    w["lp"] = np.log1p(w["precip"])
    lo, hi = CLIMATOLOGY_YEARS
    ref = w[(w["week"].dt.year >= lo) & (w["week"].dt.year <= hi)]
    clim = ref.groupby(["iso3", "woy"]).agg(t_mu=("tmean", "mean"), t_sd=("tmean", "std"),
                                            p_mu=("lp", "mean"), p_sd=("lp", "std")).reset_index()
    w = w.merge(clim, on=["iso3", "woy"], how="left").sort_values(["iso3", "week"])
    w["temp_anom"] = (w["tmean"] - w["t_mu"]) / w["t_sd"]
    w["precip_anom"] = (w["lp"] - w["p_mu"]) / w["p_sd"]
    for col in ("temp_anom", "precip_anom"):
        w[f"{col}_4w"] = w.groupby("iso3")[col].transform(
            lambda s: s.rolling(ANOMALY_WINDOW, min_periods=ANOMALY_WINDOW).mean())
    cols = ["iso3", "week", "tmean", "precip", "temp_anom", "precip_anom", *CLIMATE_PREDICTORS]
    w = w[cols].reset_index(drop=True)
    num = w.columns.drop(["iso3", "week"])
    w[num] = w[num].round(4)
    return w


def climate_matrix(climate: pd.DataFrame, iso3: str, weeks: pd.Series) -> np.ndarray:
    """Predictors for one country aligned to ``weeks`` (NaN where no climate data)."""
    c = climate[climate["iso3"] == iso3].set_index("week")[list(CLIMATE_PREDICTORS)]
    return c.reindex(pd.DatetimeIndex(weeks)).to_numpy(float)


def socioeconomic(wb: pd.DataFrame) -> pd.DataFrame:
    """Country x year table from the World Bank long table, with population density by sex."""
    t = wb.pivot_table(index=["iso3", "year"], columns="variable", values="value").reset_index()
    t.columns.name = None
    area = t.get("land_area_km2")
    for sex in ("female", "male"):
        if f"population_{sex}" in t and area is not None:
            t[f"population_density_{sex}_km2"] = t[f"population_{sex}"] / area
    return t.round(4)
