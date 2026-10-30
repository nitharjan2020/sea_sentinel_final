"""Real-time replay of the detection methods on as-of snapshots rebuilt from report dates.

``detection_eval`` judges every week on finalized data. In real time the latest weeks are
under-reported, and so is the baseline they are compared with. Here, at every week ``T`` from
``REPLAY_START`` to one year before the last panel week, each series is cut back to the records
reported by the end of ``T``, and both methods judge the last ``MAX_LAG + 1`` weeks
(``w = T - d``, d = 0..MAX_LAG) as they could have at the time. Two inputs:

* **as reported**: the counts known at T;
* **nowcast-corrected**: the last 52 weeks replaced by the nowcast mean at T, in the baseline and in
  the judged week alike. An alarm still needs at least one reported outbreak.

The reference is the alarm on finalized data at the default thresholds. For each finalized alarm the
replay gives the first ``d`` at which its week was flagged in real time; for each week flagged in
real time, whether the finalized data flag it too.

Same caveat as the nowcast backtest: snapshots replay reporting timing from today's extract, so later
edits or deletions of records are not seen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .detection import DEFAULT_THRESHOLD, alarms, nb_baseline, rolling_baseline, threshold
from .nowcast import D_MAX, nowcast

REPLAY_START = pd.Timestamp("2022-01-03")  # first as-of week: the forecasting held-out period
MAX_LAG = 12                               # weeks after the judged week that the replay follows it
LAG_MARKS = (0, 1, 4, MAX_LAG)             # "flagged within d weeks" columns of the summary
INPUTS = ("as_reported", "nowcast_adjusted")


def origins(last_week: pd.Timestamp) -> pd.DatetimeIndex:
    """Weekly as-of dates; every judged week is at least one year older than ``last_week``."""
    return pd.date_range(REPLAY_START, last_week - pd.Timedelta(weeks=D_MAX), freq="W-MON")


def asof_counts(event_pos: np.ndarray, report_week: np.ndarray, T: pd.Timestamp, P: int) -> np.ndarray:
    """Weekly counts for panel rows 0..P from records reported by the end of week ``T``.

    Rows before the first known outbreak are missing (as in the panel); later rows are reported counts,
    zero where nothing was reported.
    """
    keep = (report_week <= T) & (event_pos >= 0) & (event_pos <= P)
    y = np.bincount(event_pos[keep], minlength=P + 1).astype(float)
    first = np.flatnonzero(y > 0)
    y[: first[0] if len(first) else P + 1] = np.nan
    return y


def judge(y: np.ndarray, reported: np.ndarray, week: pd.Series, log_pop: pd.Series | None) -> dict[str, np.ndarray]:
    """Default-threshold alarms for the last MAX_LAG + 1 rows of ``y`` (index d = 0..MAX_LAG back).

    ``reported`` holds the counts reported so far: a week can alarm only if it has at least one.
    """
    P = len(y) - 1
    lags = np.arange(MAX_LAG + 1)
    pos = P - lags
    ok = pos >= 0
    pos = np.where(ok, pos, 0)
    possible = ok & (reported[pos] >= 1)  # no alarm without a report, so no fit is needed there
    out = {m: np.zeros(len(lags), dtype=bool) for m in DEFAULT_THRESHOLD}
    if not possible.any():
        return out
    ys = pd.Series(y)
    base = pd.concat([rolling_baseline(ys), nb_baseline(ys, week, log_pop, targets=pos[possible])], axis=1)
    b = base.iloc[pos]
    evaluable = ~np.isnan(y[pos]) & b["base_mean"].notna().to_numpy() & b["nb_mu"].notna().to_numpy()
    for method, level in DEFAULT_THRESHOLD.items():
        out[method] = alarms(y[pos], threshold(b, method, level)) & evaluable & possible
    return out


def replay(outbreaks: pd.DataFrame, panel: pd.DataFrame, series: list[tuple[str, str]],
           last_week: pd.Timestamp) -> pd.DataFrame:
    """One row per series x as-of week x lag d x method x input, with the real-time alarm."""
    data = {}
    for iso3, disease in series:
        s = panel[(panel["iso3"] == iso3) & (panel["disease"] == disease)].sort_values("week").reset_index(drop=True)
        o = outbreaks[(outbreaks["iso3"] == iso3) & (outbreaks["disease"] == disease)]
        pos = {w: i for i, w in enumerate(s["week"])}
        event_pos = o["event_week"].map(pos).fillna(-1).astype(int).to_numpy()
        data[(iso3, disease)] = (s, pos, event_pos, o["report_week"].to_numpy())
    rows = []
    for T in origins(last_week):
        nc = nowcast(outbreaks, T, series)
        nc_mean = {k: g.set_index("week")["nowcast_mean"] for k, g in nc.groupby(["iso3", "disease"])} if len(nc) else {}
        for key, (s, pos, event_pos, report_week) in data.items():
            if T not in pos:
                continue
            P = pos[T]
            y = asof_counts(event_pos, report_week, T, P)
            if np.isnan(y).all():
                continue
            week = s["week"].iloc[: P + 1]
            log_pop = s["log_population"].iloc[: P + 1] if "log_population" in s else None
            y_nc = y.copy()
            if key in nc_mean:  # inactive series get no nowcast: the system would use the reports
                m = nc_mean[key]
                idx = np.array([pos[w] for w in m.index if w in pos and pos[w] <= P], dtype=int)
                idx = idx[~np.isnan(y[idx])]
                y_nc[idx] = m.loc[s["week"].iloc[idx]].to_numpy()
            for inp, values in (("as_reported", y), ("nowcast_adjusted", y_nc)):
                for method, a in judge(values, y, week, log_pop).items():
                    rows.extend({"iso3": key[0], "disease": key[1], "as_of": T, "lag": d,
                                 "week": T - pd.Timedelta(weeks=int(d)), "method": method, "input": inp,
                                 "alarm": bool(a[d])} for d in range(MAX_LAG + 1))
    return pd.DataFrame(rows)


def first_alarms(rp: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """Each finalized alarm followed for MAX_LAG weeks: first lag at which it was flagged in real time."""
    weeks = rp["as_of"].drop_duplicates()
    lo, hi = weeks.min(), weeks.max() - pd.Timedelta(weeks=MAX_LAG)  # weeks seen at every lag 0..MAX_LAG
    keys = set(zip(rp["iso3"], rp["disease"]))
    b = baselines[(baselines["week"] >= lo) & (baselines["week"] <= hi)
                  & np.array([k in keys for k in zip(baselines["iso3"], baselines["disease"])])]
    final = pd.concat([b.loc[b[f"{m}_alarm"], ["iso3", "disease", "week"]].assign(method=m) for m in DEFAULT_THRESHOLD])
    hit = rp[rp["alarm"]].groupby(["iso3", "disease", "week", "method", "input"])["lag"].min()
    rows = []
    for inp in INPUTS:
        f = final.assign(input=inp)
        f["first_lag"] = hit.reindex(pd.MultiIndex.from_frame(f[hit.index.names])).to_numpy()
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def confirmed(rp: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """Each (series, week) flagged in real time within MAX_LAG weeks, and whether finalized data flag it."""
    weeks = rp["as_of"].drop_duplicates()
    lo, hi = weeks.min(), weeks.max() - pd.Timedelta(weeks=MAX_LAG)
    rt = rp[rp["alarm"] & (rp["week"] >= lo) & (rp["week"] <= hi)]
    rt = rt.groupby(["iso3", "disease", "week", "method", "input"])["lag"].min().rename("first_lag").reset_index()
    fin = baselines[["iso3", "disease", "week"] + [f"{m}_alarm" for m in DEFAULT_THRESHOLD]]
    rt = rt.merge(fin, on=["iso3", "disease", "week"], how="left")
    rt["final_alarm"] = np.select([rt["method"] == m for m in DEFAULT_THRESHOLD],
                                  [rt[f"{m}_alarm"].fillna(False).astype(bool) for m in DEFAULT_THRESHOLD], False)
    return rt.drop(columns=[f"{m}_alarm" for m in DEFAULT_THRESHOLD])


def summarize(first: pd.DataFrame, conf: pd.DataFrame) -> pd.DataFrame:
    """Per method x input x disease (and all diseases): share of finalized alarms flagged within d weeks,
    median weeks to the first real-time alarm, and the share of real-time alarms confirmed."""
    rows = []
    for disease in [None, *sorted(first["disease"].unique())]:
        f = first if disease is None else first[first["disease"] == disease]
        c = conf if disease is None else conf[conf["disease"] == disease]
        for (method, inp), g in f.groupby(["method", "input"]):
            lag = g["first_lag"].to_numpy(float)
            cc = c[(c["method"] == method) & (c["input"] == inp)]
            rows.append({
                "disease": disease or "all", "method": method, "input": inp, "final_alarms": len(g),
                **{f"flagged_within_{d}": float(np.mean(lag <= d)) if len(g) else np.nan for d in LAG_MARKS},
                "median_weeks_to_flag": float(np.median(np.where(np.isnan(lag), np.inf, lag))) if len(g) else np.nan,
                "realtime_alarms": len(cc),
                "share_confirmed": cc["final_alarm"].mean() if len(cc) else np.nan,
            })
    return pd.DataFrame(rows).round(4)


def curve(first: pd.DataFrame) -> pd.DataFrame:
    """Cumulative share of finalized alarms flagged in real time by lag d, per method and input."""
    rows = [{"method": m, "input": i, "lag": d, "share_flagged": float(np.mean(g["first_lag"].to_numpy(float) <= d))}
            for (m, i), g in first.groupby(["method", "input"]) for d in range(MAX_LAG + 1)]
    return pd.DataFrame(rows).round(4)


def run(outbreaks: pd.DataFrame, panel: pd.DataFrame, audit: pd.DataFrame, baselines: pd.DataFrame,
        last_week: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Replay for every series with at least detection-level coverage (as in ``detection_eval``)."""
    tiers = audit["Forecast eligibility"].str.match("Eligible|Conditional|Detection")
    series = list(audit.loc[tiers, ["ISO3", "Disease"]].itertuples(index=False, name=None))
    rp = replay(outbreaks, panel, series, last_week)
    first = first_alarms(rp, baselines)
    conf = confirmed(rp, baselines)
    return {"replay": rp, "first": first, "confirmed": conf, "summary": summarize(first, conf),
            "curve": curve(first)}
