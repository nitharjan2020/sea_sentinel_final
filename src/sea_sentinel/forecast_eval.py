"""Backtests and the latest forecast for the forecast-eligible series.

1. **Retrospective backtest on finalized data.** Forecasts are made at every week with at least
   ``MIN_HISTORY`` observed weeks behind it and scored against the finalized count h weeks later.
   This shows model skill if reporting were complete; it is not a real-time replay.
2. **Real-time replay.** At the nowcast backtest's as-of dates, the last 52 weeks are replaced by
   what had been reported by then, either taken at face value ("as reported") or corrected by the
   nowcast ("nowcast-adjusted"). The ensemble is then scored against the finalized count. This shows
   what a forecaster would actually have achieved, and whether nowcasting helps.
3. **Latest forecast** from the last panel week, from both inputs: the counts as reported, and the
   counts corrected by the current nowcast. The replay shows neither input is better everywhere.

4. **Climate test.** NB-AR with and without the climate predictors (``covariates.py``), scored on the
   same origins as the finalized backtest, so the comparison is paired. It is reported for the
   selection and held-out periods separately and does not change the model selection.

**Model selection without peeking.** The primary model is the one with the lowest relative CRPS on
forecasts made before ``SELECTION_END``; its skill is then reported on the later, held-out period.

Scores: CRPS, log score (floored at -10), absolute error of the median, and 80%/95% interval
coverage. Relative CRPS is the geometric mean over series of each model's CRPS divided by the naive
benchmark's (< 1 is better), so that large series do not dominate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .covariates import climate_matrix
from .forecast import HORIZONS, MIN_HISTORY, MODELS, Series, forecast_origin, pmfs, scores

SELECTION_END = pd.Timestamp("2022-01-01")


def eligible_series(audit: pd.DataFrame) -> list[tuple[str, str]]:
    e = audit[audit["Forecast eligibility"].str.startswith("Eligible")]
    return list(e[["ISO3", "Disease"]].itertuples(index=False, name=None))


def _series(panel: pd.DataFrame, iso3: str, disease: str) -> pd.DataFrame:
    return panel[(panel["iso3"] == iso3) & (panel["disease"] == disease)].sort_values("week").reset_index(drop=True)


def _score_origin(series: Series, T: int, truth: np.ndarray, models=MODELS) -> list[dict]:
    rows = []
    for h in HORIZONS:
        if T + h >= len(truth) or np.isnan(truth[T + h]):
            continue
        P = pmfs(series.params(T, h), truth[T + h])
        for m in models:
            rows.append({"horizon": h, "model": m, "truth": truth[T + h], **scores(P[m], truth[T + h])})
    return rows


# --------------------------------------------------------------------------- 1. finalized backtest
def backtest(panel: pd.DataFrame, series: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for iso3, disease in series:
        s = _series(panel, iso3, disease)
        ser = Series(s["week"], s["outbreaks"])
        observed = np.cumsum(~np.isnan(ser.yv))
        truth = ser.yv
        for T in np.flatnonzero((observed >= MIN_HISTORY) & ~np.isnan(ser.yv)):
            for r in _score_origin(ser, T, truth):
                rows.append({"iso3": iso3, "disease": disease, "origin": s.loc[T, "week"], **r})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2. real-time replay
def replay(panel: pd.DataFrame, nowcast_bt: pd.DataFrame, series: list[tuple[str, str]],
           model: str) -> pd.DataFrame:
    """``model`` scored from finalized, as-reported and nowcast-adjusted inputs at the nowcast as-of dates."""
    rows = []
    for iso3, disease in series:
        s = _series(panel, iso3, disease)
        nb = nowcast_bt[(nowcast_bt["iso3"] == iso3) & (nowcast_bt["disease"] == disease)]
        truth = s["outbreaks"].to_numpy(float)
        pos = {w: i for i, w in enumerate(s["week"])}
        for as_of, g in nb.groupby("as_of"):
            if as_of not in pos:
                continue
            T = pos[as_of]
            for variant, col in (("finalized", None), ("as_reported", "reported"), ("nowcast_adjusted", "nowcast_mean")):
                y = s["outbreaks"].iloc[: T + 1].copy()
                if col is not None:
                    recent = g.set_index("week")[col]
                    idx = [pos[w] for w in recent.index if w in pos and pos[w] <= T]
                    y.iloc[idx] = recent.loc[[s.loc[i, "week"] for i in idx]].to_numpy()
                ser = Series(s["week"].iloc[: T + 1], y)
                if np.isnan(ser.yv[T]) or np.sum(~np.isnan(ser.yv)) < MIN_HISTORY:
                    continue
                for r in _score_origin(ser, T, truth, models=(model,)):
                    rows.append({"iso3": iso3, "disease": disease, "origin": as_of, "input": variant, **r})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 3. latest forecast
def latest(panel: pd.DataFrame, nowcast_latest: pd.DataFrame, series: list[tuple[str, str]]) -> pd.DataFrame:
    """Forecasts from the last panel week; the last 52 weeks come from the nowcast table, either as
    reported so far or as the nowcast mean. Series without a nowcast (inactive) get no forecast."""
    rows = []
    for iso3, disease in series:
        s = _series(panel, iso3, disease)
        nc = nowcast_latest[(nowcast_latest["iso3"] == iso3) & (nowcast_latest["disease"] == disease)]
        if nc.empty:
            continue
        pos = {w: i for i, w in enumerate(s["week"])}
        T = len(s) - 1
        for variant, col in (("as_reported", "reported"), ("nowcast_adjusted", "nowcast_mean")):
            y = s["outbreaks"].copy()
            for w, v in zip(nc["week"], nc[col]):
                if w in pos:
                    y.iloc[pos[w]] = v
            for r in forecast_origin(Series(s["week"], y), T):
                rows.append({"iso3": iso3, "disease": disease, "input": variant, "origin": s.loc[T, "week"],
                             "target_week": s.loc[T, "week"] + pd.Timedelta(weeks=r["horizon"]),
                             "batch_reporting": bool(nc["batch_reporting"].iloc[0]), **r})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 4. climate test
CLIMATE_MODELS = ("naive", "nb_ar", "nb_ar_climate")


def climate_test(panel: pd.DataFrame, series: list[tuple[str, str]], climate: pd.DataFrame) -> pd.DataFrame:
    """NB-AR with and without climate predictors (and the naive benchmark) at every backtest origin."""
    rows = []
    for iso3, disease in series:
        s = _series(panel, iso3, disease)
        ser = Series(s["week"], s["outbreaks"], climate=climate_matrix(climate, iso3, s["week"]))
        observed = np.cumsum(~np.isnan(ser.yv))
        for T in np.flatnonzero((observed >= MIN_HISTORY) & ~np.isnan(ser.yv)):
            for r in _score_origin(ser, T, ser.yv, models=CLIMATE_MODELS):
                rows.append({"iso3": iso3, "disease": disease, "origin": s.loc[T, "week"], **r})
    ct = pd.DataFrame(rows)
    ct["period"] = np.where(ct["origin"] < SELECTION_END, "selection", "test")
    return ct


def _geo_ratio(per: pd.DataFrame, num: str, den: str) -> float:
    r = per[num] / per[den]
    r = r[(per[num] > 0) & (per[den] > 0)]
    return float(np.exp(np.mean(np.log(r)))) if len(r) else np.nan


def climate_summary(ct: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per period: relative CRPS vs naive of both NB-AR variants, the climate / no-climate CRPS ratio
    (geometric mean over series; < 1 = climate helps) and the series where climate helps; plus the
    per-series table."""
    per = ct.pivot_table(index=["period", "iso3", "disease"], columns="model", values="crps", aggfunc="mean")
    per = per.reset_index()
    per["climate_vs_nb_ar"] = per["nb_ar_climate"] / per["nb_ar"]
    rows = []
    for period, g in per.groupby("period"):
        rows.append({"period": period, "series": len(g),
                     "forecasts": int((ct["period"] == period).sum() / len(CLIMATE_MODELS)),
                     "nb_ar_relative_crps": _geo_ratio(g, "nb_ar", "naive"),
                     "nb_ar_climate_relative_crps": _geo_ratio(g, "nb_ar_climate", "naive"),
                     "climate_vs_nb_ar": _geo_ratio(g, "nb_ar_climate", "nb_ar"),
                     "series_climate_better": int((g["climate_vs_nb_ar"] < 1).sum())})
    return pd.DataFrame(rows).round(4), per.round(4)


# --------------------------------------------------------------------------- summaries
def _rel_crps(per_series: pd.DataFrame, by: list[str]) -> pd.Series:
    base = per_series[per_series["model"] == "naive"].set_index(["iso3", "disease"] + [b for b in by if b != "model"])["crps"]
    ps = per_series.join(base.rename("crps_naive"), on=base.index.names)
    ps = ps[(ps["crps"] > 0) & (ps["crps_naive"] > 0)]
    return ps.groupby(by).apply(lambda g: float(np.exp(np.mean(np.log(g["crps"] / g["crps_naive"])))),
                                include_groups=False)


def summarize(bt: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Macro-averaged scores (each series weighted equally) plus relative CRPS vs naive."""
    per_series = bt.groupby(["iso3", "disease"] + by, observed=True).agg(
        n=("crps", "size"), crps=("crps", "mean"), log_score=("log_score", "mean"),
        mae=("abs_error", "mean"), coverage80=("in80", "mean"), coverage95=("in95", "mean"),
    ).reset_index()
    out = per_series.groupby(by).agg(
        series=("iso3", "size"), forecasts=("n", "sum"), crps=("crps", "mean"), log_score=("log_score", "mean"),
        mae=("mae", "mean"), coverage80=("coverage80", "mean"), coverage95=("coverage95", "mean"),
    )
    if "model" in by:
        out["relative_crps"] = _rel_crps(per_series, by)
    return out.reset_index().round(4)


def by_series(bt: pd.DataFrame) -> pd.DataFrame:
    """Per series and model, all horizons pooled, with CRPS relative to the naive benchmark."""
    out = bt.groupby(["iso3", "disease", "model"]).agg(
        n=("crps", "size"), crps=("crps", "mean"), log_score=("log_score", "mean"), mae=("abs_error", "mean"),
        coverage80=("in80", "mean"), coverage95=("in95", "mean")).reset_index()
    naive = out[out["model"] == "naive"].set_index(["iso3", "disease"])["crps"]
    out["relative_crps"] = out["crps"] / out.join(naive.rename("n_"), on=["iso3", "disease"])["n_"]
    return out.round(4)


def replay_summary(rp: pd.DataFrame) -> pd.DataFrame:
    per = rp.groupby(["iso3", "disease", "input", "horizon"]).agg(
        crps=("crps", "mean"), mae=("abs_error", "mean"), coverage80=("in80", "mean"),
    ).reset_index()
    base = per[per["input"] == "finalized"].set_index(["iso3", "disease", "horizon"])["crps"]
    per = per.join(base.rename("crps_finalized"), on=["iso3", "disease", "horizon"])
    ok = per[(per["crps"] > 0) & (per["crps_finalized"] > 0)]
    rel = ok.groupby(["input", "horizon"]).apply(
        lambda g: float(np.exp(np.mean(np.log(g["crps"] / g["crps_finalized"])))), include_groups=False)
    out = per.groupby(["input", "horizon"]).agg(series=("iso3", "size"), crps=("crps", "mean"), mae=("mae", "mean"),
                                                 coverage80=("coverage80", "mean"))
    out["crps_vs_finalized"] = rel
    return out.reset_index().round(4)


def select_model(bt: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Primary model = lowest relative CRPS on the selection period; scores for both periods."""
    bt = bt.assign(period=np.where(bt["origin"] < SELECTION_END, "selection", "test"))
    by_period = summarize(bt, ["period", "model"])
    sel = by_period[(by_period["period"] == "selection") & (by_period["model"] != "naive")]
    return sel.loc[sel["relative_crps"].idxmin(), "model"], by_period


def run(panel: pd.DataFrame, audit: pd.DataFrame, nowcast: dict[str, pd.DataFrame],
        climate: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    series = eligible_series(audit)
    bt = backtest(panel, series)
    primary, by_period = select_model(bt)
    rp = replay(panel, nowcast["backtest"], series, primary)
    out = {}
    if climate is not None:
        ct = climate_test(panel, series, climate)
        out["climate_summary"], out["climate_by_series"] = climate_summary(ct)
    return out | {
        "primary": primary,
        "backtest": bt,
        "by_period": by_period,
        "summary": summarize(bt, ["model", "horizon"]),
        "by_series": by_series(bt),
        "by_series_test": by_series(bt[bt["origin"] >= SELECTION_END]),
        "replay": rp,
        "replay_summary": replay_summary(rp),
        "latest": latest(panel, nowcast["latest"], series),
    }
