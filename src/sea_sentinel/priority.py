"""Priority alerts for stakeholders: the pipeline's current signals combined per country x disease.

Each series with at least detection-level coverage gets one row for the latest panel week. Every
signal below that fires adds one point and is named in ``reasons``:

1. **National alarm**: either detection method alarmed in the last ``RECENT_WEEKS`` weeks
   (``detection_eval.latest_alerts``). Twelve weeks, not one, because recent weeks are incomplete:
   in the real-time replay a third of alarms appeared only after 4 weeks.
2. **Province alarm**: the subnational system alarmed (matched cut-off) in the same window; only the
   series covered by ``detection_subnational``.
3. **Reporting backlog**: the nowcast expects at least ``BACKLOG_MIN`` outbreaks in the last 4 weeks
   that have not been reported yet.
4. **Rising forecast**: the primary model's 4-week forecast (nowcast-corrected input) is at least
   ``RISE_FACTOR`` times the series' usual 4-week count (mean over the 52 weeks ending
   ``RECENT_WEEKS`` weeks ago, which are nearly complete) and at least ``RISE_MIN`` outbreaks.
5. **HPAI in wild birds** (HPAI only): wild-bird HPAI confirmed in the country in the last
   ``WILD_BIRD_WEEKS`` weeks and reported by now. These records are outside the outbreak scope
   (``non-poultry bird``) and too few to model; mammals (captive tigers, civets) are left out.

Priority: **High** = 2 or more points, **Medium** = 1, **Low** = 0.

**Confidence.** Where the nowcast flags batch reporting, signals 3 and 4 rest on an unreliable
correction, so they are listed as notes and add no point, and confidence is ``low``. Series with no
report in the nowcast window (inactive) also get ``low``.

This is a transparent decision aid built from signals evaluated elsewhere in the report, not a
validated risk score; the rules were set in advance and not tuned to WAHIS events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RECENT_WEEKS = 12
BACKLOG_MIN = 3
RISE_FACTOR = 1.5
RISE_MIN = 2
WILD_BIRD_WEEKS = 26
MAMMAL_WORDS = ("tiger", "leopard", "civet", "mammal")  # whole words would miss e.g. "pigeon" vs "pig"
PRIORITY = {0: "Low", 1: "Medium"}  # 2 or more points: High


def wild_bird_events(standardized: pd.DataFrame) -> pd.DataFrame:
    """Confirmed HPAI in wild birds (excluded from the outbreak scope), one row per date and place."""
    s = standardized[(standardized["exclude_reason"] == "non-poultry bird") & (standardized["disease"] == "HPAI")]
    animal = s["animal_type"].fillna("").astype(str).str.lower()
    s = s[animal.str.contains("wild") & ~animal.str.contains("|".join(MAMMAL_WORDS))]
    s = s.assign(event_date=pd.to_datetime(s["event_date"]), report_date=pd.to_datetime(s["report_date"]))
    g = s.groupby(["iso3", "event_date", "admin1"], dropna=False)
    return g.agg(report_date=("report_date", "min"), records=("record_id", "size"),
                 species=("animal_type", lambda x: " | ".join(sorted(set(x))))).reset_index()


def wild_bird_signal(events: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Per country: wild-bird HPAI events in the last WILD_BIRD_WEEKS weeks, known by the end of ``as_of``."""
    end = as_of + pd.Timedelta(days=6)
    e = events[(events["report_date"] <= end) & (events["event_date"] > end - pd.Timedelta(weeks=WILD_BIRD_WEEKS))]
    return e.groupby("iso3").agg(wild_bird_events=("event_date", "size"), wild_bird_last=("event_date", "max"))


def _series_list(audit: pd.DataFrame) -> list[tuple[str, str]]:
    tiers = audit["Forecast eligibility"].str.match("Eligible|Detection")
    return list(audit.loc[tiers, ["ISO3", "Disease"]].itertuples(index=False, name=None))


def build(audit: pd.DataFrame, panel: pd.DataFrame, det_latest: pd.DataFrame, sub_latest: pd.DataFrame,
          nowcast_latest: pd.DataFrame, forecast_latest: pd.DataFrame, primary: str,
          wild_birds: pd.DataFrame, end_week: pd.Timestamp) -> pd.DataFrame:
    """One row per series: signals, points, priority, confidence, reasons and notes."""
    since = end_week - pd.Timedelta(weeks=RECENT_WEEKS - 1)
    wb = wild_bird_signal(wild_birds, end_week)
    fc = forecast_latest[(forecast_latest["model"] == primary) & (forecast_latest["input"] == "nowcast_adjusted")]
    rows = []
    for iso3, disease in _series_list(audit):
        key = (det_latest["iso3"] == iso3) & (det_latest["disease"] == disease) & (det_latest["week"] >= since)
        nat = det_latest[key]
        methods = [m for m in ("rolling", "negbin") if nat[f"{m}_alarm"].any()]
        sk = (sub_latest["iso3"] == iso3) & (sub_latest["disease"] == disease) & (sub_latest["week"] >= since)
        prov = sub_latest[sk]
        nc = nowcast_latest[(nowcast_latest["iso3"] == iso3) & (nowcast_latest["disease"] == disease)
                            & (nowcast_latest["weeks_back"] < 4)]
        batch = bool(nc["batch_reporting"].any()) if len(nc) else False
        reported4, expected4 = (nc["reported"].sum(), nc["nowcast_mean"].sum()) if len(nc) else (np.nan, np.nan)
        s = panel[(panel["iso3"] == iso3) & (panel["disease"] == disease)].sort_values("week")
        past = s[(s["week"] < since) & (s["week"] >= since - pd.Timedelta(weeks=52))]["outbreaks"]
        usual4 = 4 * past.mean() if past.notna().any() else np.nan
        f = fc[(fc["iso3"] == iso3) & (fc["disease"] == disease)]
        forecast4 = f["mean"].sum() if len(f) else np.nan
        pop = s["population_head"].iloc[-1] if "population_head" in s and len(s) else np.nan

        sig = {
            "national_alarm": bool(methods),
            "province_alarm": len(prov) > 0,
            "reporting_backlog": bool(len(nc)) and expected4 - reported4 >= BACKLOG_MIN,
            "rising_forecast": bool(len(f)) and forecast4 >= max(RISE_FACTOR * np.nan_to_num(usual4), RISE_MIN),
            "wild_birds": disease == "HPAI" and iso3 in wb.index,
        }
        reasons, notes = [], []
        if sig["national_alarm"]:
            reasons.append(f"national alarm ({' + '.join('rolling' if m == 'rolling' else 'NB' for m in methods)}; "
                           f"latest week of {nat['week'].max():%d %b %Y})")
        if sig["province_alarm"]:
            top = prov.sort_values("week").iloc[-1]
            reasons.append(f"province alarm ({top['admin1']}, week of {top['week']:%d %b %Y})")
        for name, text in (("reporting_backlog", f"about {expected4 - reported4:.0f} outbreaks in the last 4 weeks "
                                                 f"not yet reported ({reported4:.0f} so far)"),
                           ("rising_forecast", f"4-week forecast {forecast4:.1f} vs usual {np.nan_to_num(usual4):.1f}")):
            if sig[name] and batch:
                sig[name] = False
                notes.append(text + ": not counted (batch reporting)")
            elif sig[name]:
                reasons.append(text)
        if sig["wild_birds"]:
            reasons.append(f"HPAI in wild birds ({int(wb.loc[iso3, 'wild_bird_events'])} events, latest "
                           f"{wb.loc[iso3, 'wild_bird_last']:%d %b %Y})")
        if not len(nc):
            notes.append("no reports in the nowcast window (inactive series)")
        points = int(sum(sig.values()))
        rows.append({
            "iso3": iso3, "disease": disease, "as_of": end_week, "priority": PRIORITY.get(points, "High"),
            "points": points, "confidence": "low" if batch or not len(nc) else "normal", **sig,
            "reported_last4": reported4, "nowcast_last4": expected4, "forecast_next4": forecast4,
            "usual_4wk": usual4, "population_head": pop, "batch_reporting": batch,
            "reasons": "; ".join(reasons), "notes": "; ".join(notes),
        })
    out = pd.DataFrame(rows).sort_values(["points", "iso3", "disease"], ascending=[False, True, True])
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(2)
    return out.reset_index(drop=True)


def wahis_check(priority: pd.DataFrame, wahis: pd.DataFrame, end_week: pd.Timestamp, weeks: int = 26) -> pd.DataFrame:
    """WAHIS events that started in the last ``weeks`` weeks, with their series' current priority
    (a partial check only: these events were not used to set the rules)."""
    w = wahis.dropna(subset=["event_start_date", "iso3", "disease"]).copy()
    w["event_start_date"] = pd.to_datetime(w["event_start_date"])
    w = w[w["event_start_date"] > end_week - pd.Timedelta(weeks=weeks)]
    p = priority[["iso3", "disease", "priority", "points"]]
    out = w[["iso3", "disease", "event_id", "reason", "event_start_date"]].merge(p, on=["iso3", "disease"], how="left")
    out["priority"] = out["priority"].fillna("not monitored")
    return out.sort_values("event_start_date", ascending=False).reset_index(drop=True)
