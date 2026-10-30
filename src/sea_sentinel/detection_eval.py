"""Evaluation of the detection baselines: false-alert rate, sensitivity and timeliness.

There is no ground truth for "no outbreak", so performance is assessed three ways:

1. **Alarm rate on real data** - share of evaluable weeks that alarm. Some of these alarms are real
   surges, so it is an *upper bound* on the false-alert rate. It is also reported excluding weeks
   near WAHIS reference events.
2. **Simulated sensitivity** (clearly labelled simulation) - a synthetic outbreak is added to the
   real series at every evaluable week in turn: total size ~ Poisson(k x baseline SD), at least 1,
   spread over 3 weeks (30/45/25%). It counts as detected if any of those 3 weeks alarms. Because the
   guard band (2 weeks) keeps those weeks out of their own baselines, the stored thresholds stay valid
   and each injection is evaluated exactly, without refitting.
3. **WAHIS reference events** - official event start dates (new or recurrent introductions). An event
   is detected if an alarm falls within [-2, +4] weeks of its start week. The same window placed at
   every evaluable week gives the *chance* hit rate, which the event hit rate should clearly beat.

Series are weighted equally (macro averages) so the long HPAI series do not dominate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .detection import DEFAULT_THRESHOLD, NB_QUANTILES, ROLLING_Z, alarms, run_detection, threshold
from .standardize import week_start

SIZE_MULTIPLIERS = (2, 3, 5, 10)
INJECT_PROFILE = np.array([0.30, 0.45, 0.25])
REF_WINDOW = (-2, 4)  # weeks around a reference event start
LEVELS = {"rolling": ROLLING_Z, "negbin": NB_QUANTILES}
WEEKS_PER_YEAR = 365.25 / 7


def _levels():
    for method, levels in LEVELS.items():
        for level in levels:
            yield method, level


def _series(det: pd.DataFrame):
    for key, s in det.groupby(["iso3", "disease"], sort=False):
        yield key, s.reset_index(drop=True)


# --------------------------------------------------------------------------- 1. alarm rate
def _ref_mask(s: pd.DataFrame, ref_weeks: list[pd.Timestamp], before: int, after: int) -> np.ndarray:
    m = np.zeros(len(s), dtype=bool)
    for w in ref_weeks:
        m |= ((s["week"] >= w + pd.Timedelta(weeks=before)) & (s["week"] <= w + pd.Timedelta(weeks=after))).to_numpy()
    return m


# --------------------------------------------------------------------------- 2. simulation
def _injections(s: pd.DataFrame, rng: np.random.Generator, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Start rows with 3 consecutive evaluable weeks, and the injected counts (n_start x 3)."""
    ev = s["evaluable"].to_numpy()
    starts = np.flatnonzero(ev[:-2] & ev[1:-1] & ev[2:])
    sd = np.maximum(s["base_sd"].to_numpy()[starts], 1.0)
    total = np.maximum(rng.poisson(k * sd), 1)
    inj = np.array([rng.multinomial(n, INJECT_PROFILE) for n in total]).reshape(len(starts), 3)
    return starts, inj


def simulate(det: pd.DataFrame, seed: int = 2026) -> pd.DataFrame:
    """Per series x method x level x k: share of injected outbreaks detected and median weeks to detection."""
    rng = np.random.default_rng(seed)
    rows = []
    for (iso3, disease), s in _series(det):
        y = s["outbreaks"].to_numpy(float)
        thr = {(m, lv): threshold(s, m, lv) for m, lv in _levels()}
        for k in SIZE_MULTIPLIERS:
            starts, inj = _injections(s, rng, k)
            if not len(starts):
                continue
            idx = starts[:, None] + np.arange(3)
            for (method, level), t in thr.items():
                hit = alarms(y[idx] + inj, t[idx])
                detected = hit.any(axis=1)
                delay = np.where(detected, hit.argmax(axis=1), np.nan)
                rows.append({
                    "iso3": iso3, "disease": disease, "method": method, "level": level, "k": k,
                    "n_injections": len(starts), "sensitivity": detected.mean(),
                    "median_weeks_to_detect": np.nanmedian(delay) if detected.any() else np.nan,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 3. reference events
def reference_events(det: pd.DataFrame, wahis: pd.DataFrame) -> pd.DataFrame:
    """WAHIS events whose start week lies within an evaluated series' evaluable period."""
    w = wahis.dropna(subset=["event_start_date", "iso3", "disease"]).copy()
    w["start_week"] = week_start(pd.to_datetime(w["event_start_date"]))
    span = det[det["evaluable"]].groupby(["iso3", "disease"])["week"].agg(["min", "max"]).reset_index()
    w = w.merge(span, on=["iso3", "disease"])
    w = w[(w["start_week"] >= w["min"]) & (w["start_week"] <= w["max"])]
    return w[["iso3", "disease", "event_id", "subtype", "reason", "event_start_date", "start_week"]].reset_index(drop=True)


def event_hits(weeks: pd.Series, a: np.ndarray, evaluable: np.ndarray,
               ev_weeks: list[pd.Timestamp]) -> tuple[int, list[int], float]:
    """Events with an alarm in their [-2, +4]-week window, lags of the first such alarm, chance hit rate."""
    before, after = REF_WINDOW
    alarm_weeks = weeks[a]
    hits, lags = 0, []
    for w in ev_weeks:
        in_win = alarm_weeks[(alarm_weeks >= w + pd.Timedelta(weeks=before))
                             & (alarm_weeks <= w + pd.Timedelta(weeks=after))]
        if len(in_win):
            hits += 1
            lags.append((in_win.min() - w).days // 7)
    # chance: same window centred on every evaluable week
    win = np.convolve(a.astype(int), np.ones(after - before + 1, dtype=int), mode="full")
    win = win[after: after + len(a)] > 0  # window [t-2, t+4] for each t
    chance = win[evaluable].mean() if evaluable.any() else np.nan
    return hits, lags, chance


def evaluate_reference(det: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Per series x method x level: events hit, chance hit rate, alarm rates (all weeks / outside events)."""
    before, after = REF_WINDOW
    rows = []
    for (iso3, disease), s in _series(det):
        ev_weeks = events.loc[(events["iso3"] == iso3) & (events["disease"] == disease), "start_week"].tolist()
        evaluable = s["evaluable"].to_numpy()
        near_ref = _ref_mask(s, ev_weeks, before, after + 4)
        y = s["outbreaks"].to_numpy(float)
        for method, level in _levels():
            a = alarms(y, threshold(s, method, level)) & evaluable
            hits, lags, chance = event_hits(s["week"], a, evaluable, ev_weeks)
            outside = evaluable & ~near_ref
            rows.append({
                "iso3": iso3, "disease": disease, "method": method, "level": level,
                "evaluable_weeks": int(evaluable.sum()),
                "alarms": int(a.sum()),
                "alarm_rate": a[evaluable].mean() if evaluable.any() else np.nan,
                "alarm_rate_outside_events": a[outside].mean() if outside.any() else np.nan,
                "n_events": len(ev_weeks),
                "events_detected": hits,
                "event_sensitivity": hits / len(ev_weeks) if ev_weeks else np.nan,
                "chance_hit_rate": chance,
                "median_lag_weeks": float(np.median(lags)) if lags else np.nan,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- summary
def summarize(ref: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    """One row per method x level: macro averages over series plus pooled WAHIS event results."""
    g = ref.groupby(["method", "level"])
    out = pd.DataFrame({
        "alarm_rate": g["alarm_rate"].mean(),
        "alarms_per_year": g["alarm_rate"].mean() * WEEKS_PER_YEAR,
        "alarm_rate_outside_events": g["alarm_rate_outside_events"].mean(),
        "wahis_events": g["n_events"].sum(),
        "wahis_event_sensitivity": g["events_detected"].sum() / g["n_events"].sum(),
        "wahis_chance_hit_rate": g.apply(
            lambda x: np.average(x["chance_hit_rate"], weights=x["n_events"]) if x["n_events"].sum() else np.nan,
            include_groups=False,
        ),
    })
    simw = sim.pivot_table(index=["method", "level"], columns="k", values="sensitivity", aggfunc="mean")
    simw.columns = [f"sim_sensitivity_k{k}" for k in simw.columns]
    delay = sim[sim["k"] == 5].groupby(["method", "level"])["median_weeks_to_detect"].mean()
    out = out.join(simw).join(delay.rename("sim_weeks_to_detect_k5"))
    out = out.reset_index()
    out["default"] = [DEFAULT_THRESHOLD[m] == lv for m, lv in zip(out["method"], out["level"])]
    return out.round(4)


def by_series(ref: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    """Default-threshold results per series and method."""
    is_default = [DEFAULT_THRESHOLD[m] == lv for m, lv in zip(ref["method"], ref["level"])]
    r = ref[np.array(is_default)].drop(columns="level")
    s = sim[(sim["k"] == 5) & np.array([DEFAULT_THRESHOLD[m] == lv for m, lv in zip(sim["method"], sim["level"])])]
    s = s[["iso3", "disease", "method", "sensitivity", "median_weeks_to_detect"]].rename(
        columns={"sensitivity": "sim_sensitivity_k5", "median_weeks_to_detect": "sim_weeks_to_detect_k5"}
    )
    return r.merge(s, on=["iso3", "disease", "method"], how="left").round(4)


# --------------------------------------------------------------------------- robustness checks
def event_sources(events: pd.DataFrame, outbreaks: pd.DataFrame) -> pd.Series:
    """Per reference event: does the EMPRES-i data in its [-2, +4]-week window include WOAH-sourced records?

    ``"WOAH-sourced"`` if any record in the window came (also) through WOAH, ``"other sources only"``
    if records exist but none through WOAH, ``"no records"`` otherwise.
    """
    before, after = REF_WINDOW
    woah = outbreaks["reporting_sources"].fillna("").str.contains("WOAH")
    out = []
    for e in events.itertuples():
        win = ((outbreaks["iso3"] == e.iso3) & (outbreaks["disease"] == e.disease)
               & (outbreaks["event_week"] >= e.start_week + pd.Timedelta(weeks=before))
               & (outbreaks["event_week"] <= e.start_week + pd.Timedelta(weeks=after)))
        out.append("no records" if not win.any() else "WOAH-sourced" if woah[win].any() else "other sources only")
    return pd.Series(out, index=events.index, name="window_sources")


def independence(det: pd.DataFrame, events: pd.DataFrame, outbreaks: pd.DataFrame) -> pd.DataFrame:
    """WAHIS events hit at the default thresholds, split by whether WOAH-sourced records were in the window.

    WAHIS events and WOAH-sourced EMPRES-i records come from the same national reports; events
    detected without any such record show that detection does not merely echo WAHIS.
    """
    ev = events.assign(window_sources=event_sources(events, outbreaks))
    rows = []
    for (iso3, disease), s in _series(det):
        e = ev[(ev["iso3"] == iso3) & (ev["disease"] == disease)]
        if e.empty:
            continue
        evaluable = s["evaluable"].to_numpy()
        for method, level in DEFAULT_THRESHOLD.items():
            a = alarms(s["outbreaks"].to_numpy(float), threshold(s, method, level)) & evaluable
            for w, src in zip(e["start_week"], e["window_sources"]):
                hit, _, chance = event_hits(s["week"], a, evaluable, [w])
                rows.append({"iso3": iso3, "disease": disease, "method": method, "window_sources": src,
                             "hit": bool(hit), "chance_hit_rate": chance})
    r = pd.DataFrame(rows)
    out = r.groupby(["method", "window_sources"]).agg(events=("hit", "size"), detected=("hit", "sum"),
                                                     sensitivity=("hit", "mean"), chance_hit_rate=("chance_hit_rate", "mean"))
    return out.reset_index().round(4)


def offset_sensitivity(panel: pd.DataFrame, series: list[tuple[str, str]], det: pd.DataFrame,
                       wahis: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Negative-binomial detection with and without the population offset, on series that have one.

    The baseline refits its level in every window, so a FAOSTAT level that is wrong by a constant
    factor changes nothing; only the offset's change over time can move alarms. Removing the offset
    altogether bounds that effect. The rolling method uses no offset and is unaffected.
    """
    has_pop = panel.groupby(["iso3", "disease"])["log_population"].apply(lambda s: s.notna().any())
    keys = [k for k in series if has_pop.get(k, False)]
    with_pop = det[[k in set(keys) for k in zip(det["iso3"], det["disease"])]].reset_index(drop=True)
    without = run_detection(panel.assign(log_population=np.nan), keys)
    rows, perf = [], []
    for label, d in (("with offset", with_pop), ("without offset", without)):
        p = summarize(evaluate_reference(d, reference_events(d, wahis)), simulate(d))
        perf.append(p[(p["method"] == "negbin") & p["default"]].assign(offset=label))
    for (iso3, disease), a in with_pop.groupby(["iso3", "disease"], sort=False):
        b = without[(without["iso3"] == iso3) & (without["disease"] == disease)]
        x, y = a["negbin_alarm"].to_numpy(), b["negbin_alarm"].to_numpy()
        rows.append({"iso3": iso3, "disease": disease, "alarms_with": int(x.sum()), "alarms_without": int(y.sum()),
                     "alarms_both": int((x & y).sum()), "weeks_differing": int((x != y).sum())})
    return {"by_series": pd.DataFrame(rows), "summary": pd.concat(perf, ignore_index=True)}


def latest_alerts(det: pd.DataFrame, weeks: int = 12) -> pd.DataFrame:
    """Alarms (either method, default thresholds) in each series' last ``weeks`` panel weeks."""
    last = det["week"].max() - pd.Timedelta(weeks=weeks - 1)
    recent = det[(det["week"] >= last) & (det["rolling_alarm"] | det["negbin_alarm"])]
    cols = ["iso3", "disease", "week", "outbreaks", "rolling_threshold", "rolling_alarm",
            "negbin_threshold", "negbin_alarm"]
    return recent[cols].sort_values(["week", "iso3"], ascending=[False, True]).reset_index(drop=True)


def run(panel: pd.DataFrame, audit: pd.DataFrame, wahis: pd.DataFrame,
        outbreaks: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Detection + evaluation for every series with at least detection-level coverage.

    With ``outbreaks`` (deduplicated records), also the WAHIS independence check and the population
    offset sensitivity.
    """
    tiers = audit["Forecast eligibility"].str.match("Eligible|Conditional|Detection")
    series = list(audit.loc[tiers, ["ISO3", "Disease"]].itertuples(index=False, name=None))
    det = run_detection(panel, series)
    events = reference_events(det, wahis)
    ref = evaluate_reference(det, events)
    sim = simulate(det)
    out = {
        "baselines": det,
        "events": events,
        "performance": summarize(ref, sim),
        "by_series": by_series(ref, sim),
        "simulation": sim,
        "latest": latest_alerts(det),
    }
    if outbreaks is not None:
        out["independence"] = independence(det, events, outbreaks)
        out["offset"] = offset_sensitivity(panel, series, det, wahis)
    return out
