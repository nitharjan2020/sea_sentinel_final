"""Subnational (admin1) detection for endemic series, compared with national detection.

A new event in one province barely moves an endemic national weekly count. For series with at least
``MIN_UNITS`` admin1 units that each have ``MIN_UNIT_OUTBREAKS`` outbreaks, both detection baselines
are fitted to every unit's weekly series, and each unit-week gets an exceedance **score**:

* rolling: z = (count - mean) / SD;
* negative binomial: -log10 P(Y >= count) under NB(mu, alpha).

An empty baseline is scored as if it held one outbreak (SD floored at 1/sqrt(window), mu at
1/window), so a first outbreak in a quiet province scores high but a larger cluster scores higher.
The **subnational system** score for a week is the highest unit score; it alarms above a cut-off.

* *Unit weeks* follow the national reporting window: outside it they are missing; inside it a unit
  with no report counts as a reported zero (the national system was reporting).
* *Monitoring* starts once a unit has ``MIN_UNIT_OUTBREAKS`` outbreaks before its baseline window
  closes (week t - guard), so units are chosen from past data only.
* *No population offset*: FAOSTAT has no stocks below national level.
* Records without admin1, or in units not monitored, still count in the national series.

**Same alarm budget.** With dozens of units, per-unit default thresholds alarm in a large share of
weeks (reported, ``unit_default_alarm_rate``). The comparison therefore fixes the alarm rate: one
cut-off per method, common to all series, is chosen so that the subnational system alarms as often
(macro average) as national detection at its default threshold ("matched"). Further cut-offs trace
the trade-off at ``BUDGETS``.

Both systems are judged on the national evaluable weeks: alarm rate, WAHIS reference events hit
within [-2, +4] weeks against chance (as in ``detection_eval``), and **simulated localized clusters**
(simulation): n outbreaks (``CLUSTER_SIZES``) added to one unit over 3 weeks (30/45/25%), at every
start week where that unit and the nation are evaluable; the national count gets the same outbreaks.
A cluster is detected if the system alarms in any of the 3 weeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .detection import (DEFAULT_THRESHOLD, GUARD_WEEKS, NB_WINDOW, ROLLING_WINDOW, alarms, nb_baseline,
                        rolling_baseline, threshold)
from .detection_eval import INJECT_PROFILE, WEEKS_PER_YEAR, _levels, event_hits

MIN_UNIT_OUTBREAKS = 20
MIN_UNITS = 5
CLUSTER_SIZES = (3, 5, 10)
BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.15)  # subnational alarm rates for the trade-off curve
SD_FLOOR = 1 / np.sqrt(ROLLING_WINDOW)
MU_FLOOR = 1 / NB_WINDOW


# --------------------------------------------------------------------------- unit series
def unit_counts(outbreaks: pd.DataFrame) -> pd.DataFrame:
    """Outbreaks per (iso3, disease, admin1); records without admin1 are left out."""
    o = outbreaks.dropna(subset=["admin1"])
    return o.groupby(["iso3", "disease", "admin1"]).size().rename("outbreaks").reset_index()


def select_series(outbreaks: pd.DataFrame, series: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Detection series with at least MIN_UNITS admin1 units of MIN_UNIT_OUTBREAKS outbreaks each."""
    u = unit_counts(outbreaks)
    big = u[u["outbreaks"] >= MIN_UNIT_OUTBREAKS].groupby(["iso3", "disease"]).size()
    return [k for k in series if big.get(k, 0) >= MIN_UNITS]


def unit_panel(outbreaks: pd.DataFrame, national: pd.DataFrame) -> pd.DataFrame:
    """Weekly counts per admin1 unit of one series, on the national series' weeks.

    ``monitored`` marks weeks whose baseline window closes after the unit's MIN_UNIT_OUTBREAKS-th
    outbreak; ``outbreaks`` is missing wherever the national count is.
    """
    weeks = national["week"].reset_index(drop=True)
    missing = national["outbreaks"].isna().to_numpy()
    frames = []
    for unit, g in outbreaks.dropna(subset=["admin1"]).groupby("admin1"):
        if len(g) < MIN_UNIT_OUTBREAKS:
            continue
        y = weeks.map(g.groupby("event_week").size()).fillna(0).to_numpy(float, copy=True)
        before = np.concatenate([[0], np.cumsum(y)[:-1]])  # outbreaks in weeks before t
        seen = np.concatenate([np.zeros(GUARD_WEEKS), before[:-GUARD_WEEKS]])  # ... before t - guard
        y[missing] = np.nan
        frames.append(pd.DataFrame({"admin1": unit, "week": weeks, "outbreaks": y,
                                    "monitored": seen >= MIN_UNIT_OUTBREAKS}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def unit_baselines(units: pd.DataFrame) -> pd.DataFrame:
    """Rolling and NB baselines per unit (no offset), fitted only on monitored weeks."""
    frames = []
    for _, u in units.groupby("admin1", sort=False):
        u = u.reset_index(drop=True)
        y = u["outbreaks"]
        targets = np.flatnonzero(u["monitored"].to_numpy() & y.notna().to_numpy())
        base = pd.concat([rolling_baseline(y), nb_baseline(y, u["week"], targets=targets)], axis=1)
        out = pd.concat([u, base], axis=1)
        out["evaluable"] = (u["monitored"] & y.notna() & out["base_mean"].notna() & out["nb_mu"].notna())
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- scores
def scores(y, b: dict[str, np.ndarray], method: str) -> np.ndarray:
    """Exceedance score of counts ``y`` against baselines ``b`` (same shape); -inf where y < 1 or no baseline."""
    y = np.asarray(y, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        if method == "rolling":
            s = (y - b["base_mean"]) / np.maximum(b["base_sd"], SD_FLOOR)
        else:
            mu = np.maximum(b["nb_mu"], MU_FLOOR)
            alpha = np.maximum(b["nb_alpha"], 1e-8)
            size = 1 / alpha
            p = np.where(b["nb_alpha"] < 1e-8, stats.poisson.sf(y - 1, mu), stats.nbinom.sf(y - 1, size, size / (size + mu)))
            s = -np.log10(np.maximum(p, 1e-300))
    return np.where((y >= 1) & b["evaluable"] & np.isfinite(s), s, -np.inf)


def _arrays(ub: pd.DataFrame, n_weeks: int) -> dict[str, np.ndarray]:
    """Unit x week arrays of counts, baselines and evaluability."""
    n_units = ub["admin1"].nunique()
    cols = ["outbreaks", "base_mean", "base_sd", "nb_mu", "nb_alpha", "evaluable"]
    return {c: ub[c].to_numpy().reshape(n_units, n_weeks) for c in cols}


def evaluate_series(nat: pd.DataFrame, ub: pd.DataFrame, events: list[pd.Timestamp],
                    rng: np.random.Generator) -> dict:
    """National results at every threshold level, and the subnational system score per week and per
    simulated cluster (thresholded later, once the common cut-off is known)."""
    n = len(nat)
    ev_nat = nat["evaluable"].to_numpy()
    y_nat = nat["outbreaks"].to_numpy(float)
    u = _arrays(ub, n)
    y_u = u["outbreaks"].astype(float)
    ev_u = u["evaluable"].astype(bool)
    u["evaluable"] = ev_u
    # every unit x start week x cluster size gets one injection, shared by both systems
    plan = []
    for i in range(len(y_u)):
        ok = ev_u[i] & ev_nat
        starts = np.flatnonzero(ok[:-2] & ok[1:-1] & ok[2:])
        if len(starts):
            idx = starts[:, None] + np.arange(3)
            plan.append((i, idx, {k: rng.multinomial(k, INJECT_PROFILE, size=len(starts)) for k in CLUSTER_SIZES}))

    national = []
    for method, level in _levels():
        thr = threshold(nat, method, level)
        a = alarms(y_nat, thr) & ev_nat
        hits, _, chance = event_hits(nat["week"], a, ev_nat, events)
        sim = {k: np.concatenate([alarms(y_nat[idx] + inj[k], thr[idx]).any(axis=1) for _, idx, inj in plan])
               if plan else np.array([]) for k in CLUSTER_SIZES}
        national.append({"method": method, "level": level, "alarm_rate": a[ev_nat].mean(),
                         "n_events": len(events), "events_detected": hits, "chance_hit_rate": chance,
                         **{f"sim_cluster_{k}": v.mean() if len(v) else np.nan for k, v in sim.items()}})

    sub = {}
    for method, level in DEFAULT_THRESHOLD.items():
        S = scores(y_u, u, method)
        top = np.sort(S, axis=0)[::-1]
        first, second = top[0], top[1] if len(top) > 1 else np.full(n, -np.inf)
        arg = S.argmax(axis=0)
        week_score = np.where(ev_nat, first, -np.inf)
        sim = {k: [] for k in CLUSTER_SIZES}
        for i, idx, inj in plan:
            others = np.where(arg == i, second, first)[idx]
            b = {c: v[i][idx] for c, v in u.items()}
            for k in CLUSTER_SIZES:
                sim[k].append(np.maximum(others, scores(y_u[i][idx] + inj[k], b, method)).max(axis=1))
        unit_default = (alarms(y_u, threshold(ub, method, level).reshape(y_u.shape)) & ev_u).any(axis=0)
        sub[method] = {"week_score": week_score,
                       "week_unit": np.array(ub["admin1"].drop_duplicates().tolist(), dtype=object)[arg],
                       "week_unit_count": y_u[arg, np.arange(n)],
                       "sim": {k: np.concatenate(v) if v else np.array([]) for k, v in sim.items()},
                       "unit_default_alarm_rate": unit_default[ev_nat].mean()}
    return {"national": national, "sub": sub, "ev": ev_nat, "weeks": nat["week"], "events": events,
            "n_units": len(y_u)}


def calibrate(results: dict, method: str, target: float) -> float:
    """Common score cut-off whose macro-average subnational alarm rate is closest to ``target``."""
    per = [np.sort(r["sub"][method]["week_score"][r["ev"]]) for r in results.values() if r["ev"].any()]
    cands = np.unique(np.concatenate(per))
    cands = cands[np.isfinite(cands)]
    # share of each series' weeks scoring above each candidate, averaged over series
    rates = np.mean([1 - np.searchsorted(s, cands, side="right") / len(s) for s in per], axis=0)
    return float(cands[np.abs(rates - target).argmin()])


def _sub_row(r: dict, method: str, cut: float) -> dict:
    s = r["sub"][method]
    a = s["week_score"] > cut
    hits, _, chance = event_hits(r["weeks"], a, r["ev"], r["events"])
    return {"method": method, "level": cut, "alarm_rate": a[r["ev"]].mean(), "n_events": len(r["events"]),
            "events_detected": hits, "chance_hit_rate": chance,
            **{f"sim_cluster_{k}": (v > cut).mean() if len(v) else np.nan for k, v in s["sim"].items()}}


def _macro(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = rows.groupby(by, sort=False)
    return pd.DataFrame({
        "series": g.size(),
        "alarm_rate": g["alarm_rate"].mean(),
        "alarms_per_year": g["alarm_rate"].mean() * WEEKS_PER_YEAR,
        "wahis_events": g["n_events"].sum(),
        "wahis_event_sensitivity": g["events_detected"].sum() / g["n_events"].sum(),
        "wahis_chance_hit_rate": g.apply(
            lambda x: np.average(x["chance_hit_rate"], weights=x["n_events"]) if x["n_events"].sum() else np.nan,
            include_groups=False),
        **{f"sim_cluster_{k}": g[f"sim_cluster_{k}"].mean() for k in CLUSTER_SIZES},
    }).reset_index()


def latest_alerts(results: dict, cuts: dict[str, float], weeks: int = 12) -> pd.DataFrame:
    """Subnational alarms (matched cut-off) in each series' last ``weeks`` weeks: week, top unit, score."""
    rows = []
    for (iso3, disease), r in results.items():
        w = r["weeks"].to_numpy()
        recent = w >= w.max() - np.timedelta64(7 * (weeks - 1), "D")
        for method, cut in cuts.items():
            s = r["sub"][method]
            for i in np.flatnonzero(recent & (s["week_score"] > cut)):
                rows.append({"iso3": iso3, "disease": disease, "week": r["weeks"].iloc[i], "method": method,
                             "admin1": s["week_unit"][i], "unit_outbreaks": s["week_unit_count"][i],
                             "score": s["week_score"][i], "cut_off": cut})
    cols = ["iso3", "disease", "week", "method", "admin1", "unit_outbreaks", "score", "cut_off"]
    out = pd.DataFrame(rows, columns=cols).sort_values(["week", "iso3", "method"], ascending=[False, True, True])
    out[["score", "cut_off"]] = out[["score", "cut_off"]].round(4)
    return out.reset_index(drop=True)


def run(outbreaks: pd.DataFrame, baselines: pd.DataFrame, events: pd.DataFrame, seed: int = 2026) -> dict:
    """Subnational detection for the qualifying series among those in the national ``baselines``."""
    rng = np.random.default_rng(seed)
    keys = list(baselines[["iso3", "disease"]].drop_duplicates().itertuples(index=False, name=None))
    series = select_series(outbreaks, keys)
    results, unit_rows = {}, []
    for iso3, disease in series:
        nat = baselines[(baselines["iso3"] == iso3) & (baselines["disease"] == disease)].reset_index(drop=True)
        o = outbreaks[(outbreaks["iso3"] == iso3) & (outbreaks["disease"] == disease)]
        ub = unit_baselines(unit_panel(o, nat))
        ev = events.loc[(events["iso3"] == iso3) & (events["disease"] == disease), "start_week"].tolist()
        results[(iso3, disease)] = evaluate_series(nat, ub, ev, rng)
        per_unit = ub.groupby("admin1", sort=False).agg(outbreaks=("outbreaks", "sum"),
                                                         monitored_weeks=("evaluable", "sum")).reset_index()
        unit_rows.append(per_unit.assign(iso3=iso3, disease=disease, share_of_national=per_unit["outbreaks"] / len(o)))

    nat_rows = pd.DataFrame([{"iso3": k[0], "disease": k[1], **x} for k, r in results.items() for x in r["national"]])
    nat_sum = _macro(nat_rows, ["method", "level"]).assign(system="national")
    nat_sum["default"] = [DEFAULT_THRESHOLD[m] == lv for m, lv in zip(nat_sum["method"], nat_sum["level"])]
    sub_rows, sub_sum, cuts = [], [], {}
    for method in DEFAULT_THRESHOLD:
        target = nat_sum.loc[(nat_sum["method"] == method) & nat_sum["default"], "alarm_rate"].iloc[0]
        for budget, label in [(target, "matched"), *[(b, f"{100 * b:g}%") for b in BUDGETS]]:
            cut = calibrate(results, method, budget)
            rows = pd.DataFrame([{"iso3": k[0], "disease": k[1], **_sub_row(r, method, cut)} for k, r in results.items()])
            sub_sum.append(_macro(rows, ["method", "level"]).assign(system="admin1", budget=label))
            if label == "matched":
                sub_rows.append(rows.assign(system="admin1"))
                cuts[method] = cut
    summary = pd.concat([nat_sum, pd.concat(sub_sum)], ignore_index=True)
    summary["default"] = summary["default"].fillna(False).astype(bool)
    summary["matched"] = summary["budget"].eq("matched")
    nat_default = nat_rows[[DEFAULT_THRESHOLD[m] == lv for m, lv in zip(nat_rows["method"], nat_rows["level"])]]
    by_series = pd.concat([nat_default.assign(system="national"), *sub_rows], ignore_index=True)
    units = pd.concat(unit_rows, ignore_index=True)[["iso3", "disease", "admin1", "outbreaks", "share_of_national",
                                                     "monitored_weeks"]]
    unit_default = pd.DataFrame([{"iso3": k[0], "disease": k[1], "method": m, "alarm_rate": s["unit_default_alarm_rate"],
                                  "n_units": r["n_units"]} for k, r in results.items() for m, s in r["sub"].items()])
    cols = ["system", "method", "level", "budget", "default", "matched"]
    summary = summary[cols + [c for c in summary.columns if c not in cols]]
    return {"series": series, "summary": summary.round(4), "by_series": by_series.round(4), "units": units.round(4),
            "unit_default": unit_default.round(4), "latest": latest_alerts(results, cuts)}
