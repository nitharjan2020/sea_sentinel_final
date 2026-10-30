"""Render the technical report from ``docs/report_template.md`` with numbers computed by the pipeline.

The template contains the prose. Every number, country list and results table is a ``{{key}}``
placeholder filled from the pipeline's results, so the report cannot drift from the outputs.
Rendering is strict: an unknown placeholder, or a dedup review older than the current deduplication,
stops the run instead of producing a silently stale report.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, detection, detection_eval, nowcast, nowcast_eval
from .config import COUNTRIES, PROJECT_ROOT, TABLE_DIR

TEMPLATE = PROJECT_ROOT / "docs" / "report_template.md"
OUTPUT = PROJECT_ROOT / "docs" / "technical_report.md"
PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
DISEASE_LABEL = {"HPAI": "HPAI", "ASF": "ASF"}
METHOD_NAME = {"rolling": "rolling", "negbin": "negative-binomial"}
MATERIAL_CRPS = 0.02  # smaller absolute CRPS changes are reported as "no material change"


# --------------------------------------------------------------------------- formatting
def num(x: float, nd: int = 0) -> str:
    """Thousands-separated number with ``nd`` decimals."""
    return f"{x:,.{nd}f}"


def signed(x: float, nd: int = 2) -> str:
    """Signed number with a typographic minus, e.g. +0.09 / −0.20."""
    s = f"{x:+.{nd}f}"
    return s.replace("-", "−")


def half_up(x: float, nd: int = 0) -> str:
    """Round half away from zero (Python's round() rounds half to even)."""
    f = 10 ** nd
    v = math.floor(abs(x) * f + 0.5) / f
    return f"{math.copysign(v, x) if v else 0.0:.{nd}f}"


def pct(x: float, nd: int = 0) -> str:
    return half_up(100 * x, nd) + "%"


def pct_range(values, nd: int = 0) -> str:
    """'48–52%' rather than '48%–52%'."""
    lo, hi = half_up(100 * min(values), nd), half_up(100 * max(values), nd)
    return f"{lo}%" if lo == hi else f"{lo}–{hi}%"


def series_name(iso3: str, disease: str) -> str:
    return f"{COUNTRIES[iso3]} {DISEASE_LABEL[disease]}"


def join_words(items: list[str]) -> str:
    items = list(items)
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def num_range(values, fmt) -> str:
    lo, hi = min(values), max(values)
    return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)}–{fmt(hi)}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# --------------------------------------------------------------------------- rendering
def render(template: str, values: dict[str, object]) -> str:
    missing = sorted({k for k in PLACEHOLDER.findall(template) if k not in values})
    if missing:
        raise KeyError(f"Report template uses undefined values: {', '.join(missing)}")
    return PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)


# --------------------------------------------------------------------------- value sections
def _data_values(ctx: dict) -> dict:
    std, out, manifest, wahis, stocks = ctx["standardized"], ctx["outbreaks"], ctx["manifest"], ctx["wahis"], ctx["stocks"]
    retrieved = manifest.drop_duplicates("source_id").set_index("source_id")["retrieved"]
    obs = std["observation_date"].dropna()
    return {
        "extraction_date": f"{ctx['extraction']:%d %b %Y}",
        "end_week": f"{ctx['end_week']:%d %b %Y}",
        "n_empres_records": num(len(std)),
        "empres_years": f"{obs.min():%Y}–{obs.max():%Y}",
        "n_wahis_events": num(len(wahis)),
        "n_woah_sourced": num((std["reporting_source"] == "WOAH (former OIE)").sum()),
        "faostat_years": f"{stocks['year'].min()}–{stocks['year'].max()}",
        "faostat_last_year": str(stocks["year"].max()),
        "retrieved_src001": retrieved["SRC001"],
        "retrieved_src003": retrieved["SRC003"],
        "retrieved_src004": retrieved["SRC004"],
        "retrieved_src005": retrieved["SRC005"],
        "retrieved_src007": retrieved["SRC007"],
        "retrieved_src011": retrieved["SRC011"],
        "retrieved_src012": retrieved["SRC012"],
        "n_repaired": num(ctx["n_repaired"]),
        "n_in_scope": num(std["include"].sum()),
        "n_outbreaks": num(len(out)),
        **{f"n_outbreaks_{d.lower()}": num((out["disease"] == d).sum()) for d in config.DISEASES},
        **{f"obs_dated_{d.lower()}": pct((out.loc[out["disease"] == d, "date_basis"] == "observation").mean())
           for d in config.DISEASES},
        "long_gap_weeks": str(config.LONG_GAP_WEEKS),
    }


SCOPE_LABEL = {
    "included": "Included",
    "not confirmed": "Not confirmed (suspected or denied)",
    "non-domestic swine": "Swine that are not domestic",
    "low pathogenicity serotype": "Low-pathogenicity serotype",
    "non-poultry bird": "Birds that are not poultry",
    "country outside scope": "Country outside scope",
    "disease outside scope": "Disease outside scope (rabies)",
    "no usable date": "No usable date",
}


def _scope_values(ctx: dict) -> dict:
    std, out = ctx["standardized"], ctx["outbreaks"]
    counts = std["exclude_reason"].fillna("included").value_counts()
    rows = [[SCOPE_LABEL.get(k, k), num(v)] for k, v in counts.items()]
    big = out.groupby(["iso3", "disease"]).filter(lambda g: len(g) >= 100)
    share = big.groupby(["iso3", "disease"])["date_basis"].apply(lambda s: (s == "observation").mean())
    worst = share.idxmin()
    return {
        "table_scope_filter": md_table(["Reason", "Records"], rows),
        "lowest_obs_dated_series": series_name(*worst),
        "lowest_obs_dated_share": pct(share.min()),
        "n_rabies_records": num((std["disease_raw"] == "Rabies").sum()),
    }


def _dedup_values(ctx: dict) -> dict:
    lag = pd.read_csv(TABLE_DIR / "dedup_lag_profile.csv")
    dist = pd.read_csv(TABLE_DIR / "dedup_distance_profile.csv")
    grid = pd.read_csv(TABLE_DIR / "dedup_sensitivity.csv")
    summary = pd.read_csv(TABLE_DIR / "dedup_summary.csv").set_index("stage")["records"]
    if summary["after cross-source repeats"] != len(ctx["outbreaks"]):
        raise RuntimeError("The dedup review tables are out of date. Run: python run_pipeline.py --review-dedup")
    km, days = config.DEDUP_MAX_KM, config.DEDUP_WINDOW_DAYS
    r = lag.set_index("upper")["cross_to_same_ratio"]
    plateau = lag.loc[lag["lower"] >= days, "cross_to_same_ratio"].iloc[:3].mean()
    d = dist.set_index("upper")["cross_to_same_ratio"]
    beyond = dist.loc[dist["lower"] >= km, "cross_to_same_ratio"]
    return {
        "dedup_max_km": f"{km:g}",
        "dedup_window_days": str(days),
        "dedup_after_exact": num(summary["after exact repeats"]),
        "lag_ratio_d0": f"{r[0]:.2f}", "lag_ratio_d1": f"{r[1]:.2f}", "lag_ratio_d3": f"{r[3]:.2f}",
        "lag_ratio_plateau": f"{plateau:.2f}",
        "dist_ratio_05": f"{d[0.5]:.2f}", "dist_ratio_1": f"{d[1.0]:.2f}",
        "dist_ratio_beyond": num_range(beyond, lambda v: f"{v:.2f}"),
        "dedup_grid_km": f"{grid['max_km'].min():g}–{grid['max_km'].max():g} km",
        "dedup_grid_days": f"{grid['window_days'].min()}–{grid['window_days'].max()} days",
        "dedup_grid_range": f"{signed(grid['pct_vs_chosen'].min(), 1)}% to {signed(grid['pct_vs_chosen'].max(), 1)}%",
    }


TIER_ORDER = [
    ("Eligible: forecast + detection", "Forecast + detection"),
    ("Detection / descriptive only", "Detection / descriptive only"),
    ("Descriptive only (sparse)", "Descriptive only (sparse)"),
    ("Withhold: no dated records", "No records"),
]


def _coverage_values(ctx: dict) -> dict:
    audit, end_week = ctx["audit"], ctx["end_week"]
    audit = audit.assign(last=pd.to_datetime(audit["Last date"]))
    stale = audit[(audit["Forecast eligibility"].str.startswith("Eligible"))
                  & (audit["last"] < end_week - pd.Timedelta(weeks=104))]
    stale_keys = set(zip(stale["ISO3"], stale["Disease"]))
    rows = []
    for full, label in TIER_ORDER:
        t = audit[audit["Forecast eligibility"] == full]
        parts = []
        for dis in config.DISEASES:
            codes = [c + ("*" if (c, dis) in stale_keys else "") for c in t.loc[t["Disease"] == dis, "ISO3"]]
            if codes:
                parts.append(f"{DISEASE_LABEL[dis]}: {', '.join(codes)}")
        rows.append([label, "; ".join(parts) or "—", str(len(t))])
    notes = [f"\\*{series_name(r.ISO3, r.Disease)} data stop in {r.last:%B %Y}, so forecasts for it can only be "
             "retrospective." for r in stale.itertuples()]
    elig = audit[audit["Forecast eligibility"].str.startswith("Eligible")]
    return {
        "table_tiers": md_table(["Tier", "Series", "n"], rows),
        "stale_eligible_note": "\n\n".join(notes),
        "n_series": str(len(audit)),
        "n_eligible": str(len(elig)),
        "eligible_hpai_countries": join_words([COUNTRIES[c] for c in elig.loc[elig["Disease"] == "HPAI", "ISO3"]]),
        "eligible_asf_countries": join_words([COUNTRIES[c] for c in elig.loc[elig["Disease"] == "ASF", "ISO3"]]),
        "n_no_records": str((audit["Forecast eligibility"] == "Withhold: no dated records").sum()),
        "min_dated_outbreaks": str(config.ELIGIBILITY["min_dated_outbreaks"]),
        "min_active_weeks": str(config.ELIGIBILITY["min_active_weeks"]),
        "min_nonzero_weeks": str(config.ELIGIBILITY["min_nonzero_weeks"]),
    }


def _delay_values(ctx: dict) -> dict:
    out = ctx["outbreaks"]
    ok = out[out["delay_quality"] == "ok"].groupby("disease")["report_delay_days"]
    rows, med = [], {}
    for dis in config.DISEASES:
        s = ok.get_group(dis).astype(float)
        med[dis] = s.median()
        rows.append([DISEASE_LABEL[dis], num(len(s)), num(s.median()), num(s.quantile(0.9))])
    ds = ctx["delays"]
    long_ = ds[(ds["n_with_delay"] >= 30) & (ds["median_days"] > 90)].sort_values("median_days", ascending=False)
    zero = ds[(ds["n_with_delay"] >= 100) & (ds["pct_zero_delay"] > 50)]
    return {
        "table_delays": md_table(["Disease", "Outbreaks with both dates", "Median delay (days)",
                                  "90th percentile (days)"], rows),
        **{f"delay_median_{d.lower()}": num(med[d]) for d in config.DISEASES},
        "long_delay_series": join_words([f"{series_name(r.iso3, r.disease)} (median {r.median_days:.0f} days)"
                                         for r in long_.itertuples()]),
        "zero_delay_series": join_words([f"{series_name(r.iso3, r.disease)} ({r.pct_zero_delay:.0f}% same-day)"
                                         for r in zero.itertuples()]),
    }


def _recovery(base: pd.DataFrame, eligible: set) -> str:
    """Weeks for each detection threshold to return near its pre-peak level after the largest recent peak."""
    recent = base[base["evaluable"] & (base["week"] >= base["week"].max() - pd.Timedelta(weeks=156))]
    best = None
    for (iso3, dis), s in recent.groupby(["iso3", "disease"]):
        if (iso3, dis) not in eligible or s["outbreaks"].max() < 20:
            continue
        i = s["outbreaks"].idxmax()
        ratio = s.loc[i, "outbreaks"] / max(s.loc[i, "base_mean"], 1)
        if best is None or ratio > best[0]:
            best = (ratio, iso3, dis, i)
    if best is None:
        return ""
    _, iso3, dis, i = best
    s = base[(base["iso3"] == iso3) & (base["disease"] == dis)]
    peak_week = s.loc[i, "week"]
    pos = s.index.get_loc(i)
    parts = []
    for m in ("rolling", "negbin"):
        thr = s[f"{m}_threshold"].to_numpy()
        pre = np.nanmedian(thr[max(0, pos - 12):pos])
        after = np.flatnonzero(thr[pos + 1:] <= 1.25 * pre)
        parts.append(f"the {METHOD_NAME[m]} threshold "
                     + (f"took {after[0] + 1} weeks" if len(after) else f"had not recovered after {len(thr) - pos - 1} weeks"))
    return (f"After {series_name(iso3, dis)}'s peak in {peak_week:%B %Y}, {parts[1]} to return to within 25% of its "
            f"pre-peak level, while {parts[0]}. Past epidemics inside the rolling 52-week window inflate its SD.")


def _detection_values(ctx: dict) -> dict:
    det = ctx["detection"]
    perf = det["performance"][det["performance"]["default"]].set_index("method")
    rows = []
    for m, label in (("rolling", f"Rolling z = {detection.DEFAULT_THRESHOLD['rolling']:g}"),
                     ("negbin", f"Negative binomial q = {detection.DEFAULT_THRESHOLD['negbin']:g}")):
        x = perf.loc[m]
        rows.append([label, pct(x["alarm_rate"], 1), f"{x['alarms_per_year']:.1f}",
                     f"{pct(x['wahis_event_sensitivity'])} ({pct(x['wahis_chance_hit_rate'])})",
                     " / ".join(pct(x[f"sim_sensitivity_k{k}"]) for k in (2, 5, 10))])
    ratio = perf["wahis_event_sensitivity"] / perf["wahis_chance_hit_rate"]
    best = perf["wahis_event_sensitivity"].idxmax()
    other = "negbin" if best == "rolling" else "rolling"
    bys = det["by_series"]
    r = bys[(bys["method"] == "rolling") & (bys["n_events"] >= 8)].sort_values("event_sensitivity")
    fmt_ev = lambda t: f"{series_name(t.iso3, t.disease)} {t.events_detected} of {t.n_events}"  # noqa: E731
    top = [fmt_ev(t) for t in r.iloc[::-1].head(3).itertuples()]
    worst = [fmt_ev(t) for t in r.head(1).itertuples()]
    latest = det["latest"]
    both = latest[latest["rolling_alarm"] & latest["negbin_alarm"]]
    current = [f"{series_name(k[0], k[1])} (first alarm in the week of {g['week'].min():%d %b %Y})"
               for k, g in both.groupby(["iso3", "disease"])]
    elig = set(zip(*[ctx["audit"].loc[ctx["audit"]["Forecast eligibility"].str.startswith("Eligible"), c]
                     for c in ("ISO3", "Disease")]))
    return {
        "table_detection": md_table(["Default threshold", "Alarm rate", "Alarms per series-year",
                                     "WAHIS events hit (chance)", "Simulated sensitivity, k = 2 / 5 / 10"], rows),
        "n_detection_series": str(bys["iso3"].str.cat(bys["disease"]).nunique()),
        "n_ref_events": num(len(det["events"])),
        "guard_weeks": str(detection.GUARD_WEEKS),
        "rolling_window": str(detection.ROLLING_WINDOW),
        "rolling_min_obs": str(detection.ROLLING_MIN_OBS),
        "nb_window": str(detection.NB_WINDOW),
        "reweight_cutoff": f"{detection.REWEIGHT_CUTOFF:g}",
        "size_multipliers": join_words([str(k) for k in detection_eval.SIZE_MULTIPLIERS]).replace(" and ", " or "),
        "inject_profile": "/".join(f"{100 * p:.0f}" for p in detection_eval.INJECT_PROFILE) + "%",
        "ref_before": str(-detection_eval.REF_WINDOW[0]),
        "ref_after": str(detection_eval.REF_WINDOW[1]),
        "alarms_per_year_range": num_range(perf["alarms_per_year"], lambda v: f"{v:.1f}"),
        "wahis_hit_range": pct_range(perf["wahis_event_sensitivity"]),
        "wahis_chance_range": pct_range(perf["wahis_chance_hit_rate"]),
        "sim_k5_range": pct_range(perf["sim_sensitivity_k5"]),
        "wahis_vs_chance": num_range(ratio, lambda v: f"{v:.1f}"),
        "wahis_best_method": METHOD_NAME[best],
        "wahis_best": pct(perf.loc[best, "wahis_event_sensitivity"]),
        "wahis_other": pct(perf.loc[other, "wahis_event_sensitivity"]),
        "sim_k5_rolling": pct(perf.loc["rolling", "sim_sensitivity_k5"]),
        "sim_k5_negbin": pct(perf.loc["negbin", "sim_sensitivity_k5"]),
        "detection_recovery": _recovery(det["baselines"], elig),
        "detection_best_series": join_words(top),
        "detection_worst_series": join_words(worst),
        "detection_current": (f"both methods alarm for {join_words(current)}" if current
                              else "no series alarms with both methods in the last 12 weeks"),
    }


INPUT_LABEL = {"as_reported": "As reported", "nowcast_adjusted": "Nowcast-corrected"}


def _realtime_values(ctx: dict) -> dict:
    from . import detection_realtime as drt

    rt = ctx["detection"]["realtime"]
    s = rt["summary"]
    allr = s[s["disease"] == "all"].set_index(["method", "input"])
    rows = []
    for m in ("rolling", "negbin"):
        for i in drt.INPUTS:
            x = allr.loc[(m, i)]
            rows.append([METHOD_NAME[m].capitalize(), INPUT_LABEL[i], num(x["final_alarms"]),
                         " / ".join(pct(x[f"flagged_within_{d}"]) for d in drt.LAG_MARKS),
                         num(x["realtime_alarms"]), pct(x["share_confirmed"])])
    by = s[s["disease"] != "all"].set_index(["disease", "method", "input"])

    def rng(col, inp, disease=None):
        src = allr.xs(inp, level="input") if disease is None else by.loc[disease].xs(inp, level="input")
        return pct_range(src[col])

    fin = ctx["detection"]["realtime"]["first"]
    return {
        "table_realtime": md_table(["Method", "Input", "Finalized alarms",
                                    "Flagged within " + " / ".join(str(d) for d in drt.LAG_MARKS) + " weeks",
                                    "Real-time alarms", "Confirmed"], rows),
        "rt_start": f"{drt.REPLAY_START:%B %Y}",
        "rt_n_origins": num(rt["replay"]["as_of"].nunique()),
        "rt_max_lag": str(drt.MAX_LAG),
        "rt_n_final_alarms": num(fin[fin["input"] == "as_reported"].groupby("method").size().sum()),
        "rt_within0_reported": rng("flagged_within_0", "as_reported"),
        "rt_within4_reported": rng("flagged_within_4", "as_reported"),
        "rt_within4_nowcast": rng("flagged_within_4", "nowcast_adjusted"),
        "rt_never_reported": pct_range(1 - allr.xs("as_reported", level="input")[f"flagged_within_{drt.MAX_LAG}"]),
        "rt_confirmed_reported": rng("share_confirmed", "as_reported"),
        "rt_confirmed_nowcast": rng("share_confirmed", "nowcast_adjusted"),
        "rt_within4_hpai": rng("flagged_within_4", "as_reported", "HPAI"),
        "rt_within4_asf": rng("flagged_within_4", "as_reported", "ASF"),
    }


def _subnational_values(ctx: dict) -> dict:
    from . import detection_subnational as dsn

    sub = ctx["detection"]["subnational"]
    s = sub["summary"]
    nat = s[(s["system"] == "national") & s["default"]].set_index("method")
    adm = s[(s["system"] == "admin1") & s["matched"]].set_index("method")
    rows = []
    for m in ("rolling", "negbin"):
        for label, x in (("National, default threshold", nat.loc[m]), ("Admin1, same alarm rate", adm.loc[m])):
            rows.append([METHOD_NAME[m].capitalize(), label, pct(x["alarm_rate"], 1),
                         f"{pct(x['wahis_event_sensitivity'])} ({pct(x['wahis_chance_hit_rate'])})",
                         " / ".join(pct(x[f"sim_cluster_{k}"]) for k in dsn.CLUSTER_SIZES)])
    ud = sub["unit_default"].groupby("method")["alarm_rate"].mean()
    big = dsn.CLUSTER_SIZES[-1]
    return {
        "table_subnational": md_table(["Method", "System", "Alarm rate", "WAHIS events hit (chance)",
                                       "Simulated cluster of " + " / ".join(map(str, dsn.CLUSTER_SIZES))
                                       + " outbreaks"], rows),
        "sub_series": join_words([series_name(*k) for k in sub["series"]]),
        "sub_n_series": str(len(sub["series"])),
        "sub_n_units": num(len(sub["units"])),
        "sub_unit_share": pct_range(sub["units"].groupby(["iso3", "disease"])["share_of_national"].sum()),
        "sub_min_unit": str(dsn.MIN_UNIT_OUTBREAKS),
        "sub_min_units": str(dsn.MIN_UNITS),
        "sub_cluster_sizes": join_words(list(map(str, dsn.CLUSTER_SIZES))),
        "sub_big_cluster": str(big),
        "sub_unit_default_range": pct_range(ud),
        "sub_rolling_nat5": pct(nat.loc["rolling", "sim_cluster_5"]),
        "sub_rolling_adm5": pct(adm.loc["rolling", "sim_cluster_5"]),
        "sub_negbin_nat5": pct(nat.loc["negbin", "sim_cluster_5"]),
        "sub_negbin_adm5": pct(adm.loc["negbin", "sim_cluster_5"]),
        "sub_negbin_nat_big": pct(nat.loc["negbin", f"sim_cluster_{big}"]),
        "sub_negbin_adm_big": pct(adm.loc["negbin", f"sim_cluster_{big}"]),
        "sub_wahis_nat": pct_range(nat["wahis_event_sensitivity"]),
        "sub_wahis_adm": pct_range(adm["wahis_event_sensitivity"]),
        "sub_wahis_events": num(nat["wahis_events"].iloc[0]),
    }


def _nowcast_values(ctx: dict) -> dict:
    nc, out = ctx["nowcast"], ctx["outbreaks"]
    summ = nc["summary"]
    recent = summ[(~summ["batch_reporting"]) & (summ["horizon"] == "0-3")].set_index("disease")
    rows = [[DISEASE_LABEL[d], signed(x["bias_naive"]), signed(x["bias_nowcast"]),
             f"{signed(x['crps_skill_pct'], 0)}%", f"{pct(x['coverage80'])} / {pct(x['coverage95'])}"]
            for d, x in recent.iterrows()]
    nb = summ[~summ["batch_reporting"]]
    old = nb[nb["horizon"].isin(["13-26", "27-51"])]
    cov_min = nb["coverage80"].min()
    bys = nc["by_series"]
    good = bys[(~bys["batch_reporting"]) & bys["crps_skill_pct"].notna()]
    good = good.reindex(good["bias_naive"].abs().sort_values(ascending=False).index).head(4)
    ex_rows = [[series_name(r.iso3, r.disease), signed(r.bias_naive), signed(r.bias_nowcast),
                f"{signed(r.crps_skill_pct, 0)}%"] for r in good.itertuples()]
    batch = bys[bys["batch_reporting"] & (bys["n"] >= 100)]
    batch_cov = batch.sort_values("coverage80").head(1)
    over_batch = batch[batch["crps_skill_pct"] < 0].sort_values("crps_skill_pct")
    nonbatch = bys[~bys["batch_reporting"]]
    worse = nonbatch[(nonbatch["crps_nowcast"] - nonbatch["crps_naive"]) > 0.05]  # materially worse than naive
    first_year = out.groupby(["iso3", "disease"])["event_date"].min().dt.year
    latest = nc["latest"]
    lat = latest.drop_duplicates(["iso3", "disease"]).set_index(["iso3", "disease"])["batch_concentration"]
    top_key = lat.idxmax()
    within = nowcast.within_target(out)
    g = within[(within["iso3"] == top_key[0]) & (within["disease"] == top_key[1])
               & (within["event_week"] > ctx["end_week"] - pd.Timedelta(weeks=nowcast.TRAIN_WEEKS + nowcast.D_MAX))]
    per_week = g.groupby("report_week").size().sort_values(ascending=False)
    bt = nc["backtest"]
    det_latest = ctx["detection"]["latest"]
    both = det_latest[det_latest["rolling_alarm"] & det_latest["negbin_alarm"]]
    picture = []
    for (iso3, dis), _ in both.groupby(["iso3", "disease"]):
        s = latest[(latest["iso3"] == iso3) & (latest["disease"] == dis) & (latest["weeks_back"] < 4)]
        if s.empty:
            continue
        rep, exp = s["reported"].sum(), s["nowcast_mean"].sum()
        name = series_name(iso3, dis)
        if exp - rep >= 1:
            picture.append(f"In {name}, {rep:.0f} outbreaks have been reported for the last 4 weeks, but the "
                           f"nowcast expects about {exp:.0f} once reporting catches up (80% interval up to "
                           f"{s['q900'].max():.0f} in a single week).")
        else:
            picture.append(f"In {name}, {rep:.0f} outbreaks have been reported for the last 4 weeks, and few more "
                           f"are expected (nowcast about {exp:.0f}).")
    return {
        "table_nowcast": md_table(["Last 4 weeks, series without batch reporting", "Naive bias", "Nowcast bias",
                                   "CRPS improvement vs naive", "80% / 95% coverage"], rows),
        "table_nowcast_examples": md_table(["Series", "Naive bias", "Nowcast bias", "CRPS improvement"], ex_rows),
        "nowcast_summary_bias": (f"{DISEASE_LABEL[recent['bias_naive'].idxmin()]} bias "
                                 f"{signed(recent['bias_naive'].min())} → "
                                 f"{signed(recent.loc[recent['bias_naive'].idxmin(), 'bias_nowcast'])}"),
        "nowcast_crps_range": num_range(recent["crps_skill_pct"], lambda v: f"{v:.0f}") + "%",
        "nowcast_interval_word": "conservative" if cov_min >= 0.8 else "too narrow at some horizons",
        "nowcast_cov80_range": pct_range(nb["coverage80"]),
        "nowcast_old_crps_diff": num_range((old["crps_nowcast"] - old["crps_naive"]).abs(), lambda v: f"{v:.3f}"),
        "d_max": str(nowcast.D_MAX),
        "train_weeks": str(nowcast.TRAIN_WEEKS),
        "min_triangle": str(nowcast.MIN_TRIANGLE),
        "prior_weeks": str(nowcast.PRIOR_WEEKS),
        "prior_min_f": f"{nowcast.PRIOR_MIN_F:g}",
        "batch_flag": f"{nowcast.BATCH_FLAG:g}",
        "n_beyond_target": num((nowcast.delay_weeks(out) > nowcast.D_MAX).sum()),
        "batch_top_series": series_name(*top_key),
        "batch_top_share": pct(per_week.head(10).sum() / per_week.sum()),
        "batch_top_max_week": num(per_week.iloc[0]),
        "n_origins": str(bt["as_of"].nunique()),
        "origin_step": str(nowcast_eval.ORIGIN_STEP_WEEKS),
        "first_origin": f"{bt['as_of'].min():%B %Y}",
        "last_origin": f"{bt['as_of'].max():%B %Y}",
        "n_backtest_rows": num(len(bt)),
        "batch_worst": join_words([f"{series_name(r.iso3, r.disease)} (80% coverage {pct(r.coverage80)})"
                                   for r in batch_cov.itertuples()]),
        "batch_overcorrect": join_words([series_name(r.iso3, r.disease) for r in over_batch.itertuples()]),
        "nowcast_not_helping": (
            "**Where it does not help without batch reporting:** "
            + join_words([f"{series_name(r.iso3, r.disease)} (CRPS {signed(r.crps_skill_pct, 0)}%, series begins "
                          f"{first_year[(r.iso3, r.disease)]})" for r in worse.itertuples()])
            + ". Short series borrow the pooled delays of their disease, which can be longer."
            if len(worse) else
            "**Without batch reporting**, the nowcast is never materially worse than the naive count "
            "(CRPS difference at most 0.05)."),
        "nowcast_current": " ".join(picture) if picture else
                           "No series currently has alarms from both detection methods.",
    }


def _forecast_values(ctx: dict) -> dict:
    from . import forecast as fcm
    from . import forecast_eval as fem

    fc = ctx["forecast"]
    primary = fc["primary"]
    label = fcm.MODEL_LABEL
    summ = fc["summary"]
    rows = []
    for m in fcm.MODELS:
        s = summ[summ["model"] == m].sort_values("horizon")
        rows.append([label[m] + (" (primary)" if m == primary else "")]
                    + [f"{v:.2f}" for v in s["relative_crps"]]
                    + [f"{pct(s['coverage80'].mean())} / {pct(s['coverage95'].mean())}"])
    bp = fc["by_period"].set_index(["period", "model"])
    sel_rows = [[label[m] + (" (primary)" if m == primary else ""),
                 f"{bp.loc[('selection', m), 'relative_crps']:.2f}", f"{bp.loc[('test', m), 'relative_crps']:.2f}",
                 f"{pct(bp.loc[('test', m), 'coverage80'])} / {pct(bp.loc[('test', m), 'coverage95'])}"]
                for m in fcm.MODELS]
    test = fc["by_series_test"]
    tp = test[test["model"] == primary]
    losers = tp[tp["relative_crps"] >= 1]
    worst_model = summ.groupby("model")["relative_crps"].mean().idxmax()
    worst = summ[summ["model"] == worst_model]["relative_crps"]
    ens_test = bp.loc[("test", "ensemble"), "relative_crps"]

    rs = fc["replay_summary"]
    mean_crps = rs.groupby("input")["crps"].mean()
    rp = fc["replay"].pivot_table(index=["iso3", "disease"], columns="input", values="crps", aggfunc="mean")
    change = (rp["nowcast_adjusted"] / rp["as_reported"] - 1).dropna()
    material = ((rp["nowcast_adjusted"] - rp["as_reported"]).abs() >= MATERIAL_CRPS) & (change.abs() >= 0.02)
    helps = change[(change < 0) & material].sort_values()
    hurts = change[(change > 0) & material].sort_values(ascending=False)
    fmt_change = lambda s: join_words([f"{series_name(*k)} ({signed(100 * v, 0)}%)" for k, v in s.items()])  # noqa: E731
    input_label = {"finalized": "Finalized data (not available in real time)",
                   "as_reported": "As reported at the time", "nowcast_adjusted": "Nowcast-corrected"}
    replay_rows = [[input_label[i], f"{mean_crps[i]:.2f}", f"{mean_crps[i] / mean_crps['finalized']:.2f}",
                    pct(rs[rs["input"] == i]["coverage80"].mean())]
                   for i in ("finalized", "as_reported", "nowcast_adjusted")]

    lt = fc["latest"]
    lt = lt[(lt["model"] == primary) & (lt["input"] == "nowcast_adjusted")]
    latest_rows = []
    for (iso3, dis), g in lt.groupby(["iso3", "disease"], sort=False):
        g = g.sort_values("horizon")
        h1, h4 = g.iloc[0], g.iloc[-1]
        latest_rows.append([series_name(iso3, dis), f"{h1['q500']} ({h1['q100']}–{h1['q900']})",
                            f"{h4['q500']} ({h4['q100']}–{h4['q900']})", "yes" if g["batch_reporting"].any() else ""])
    origin = pd.Timestamp(lt["origin"].max()) if len(lt) else ctx["end_week"]
    no_forecast = sorted(set(fem.eligible_series(ctx["audit"])) - set(zip(lt["iso3"], lt["disease"])))
    return {
        "table_forecast_models": md_table(["Model", "h = 1", "h = 2", "h = 3", "h = 4", "80% / 95% coverage"], rows),
        "table_forecast_selection": md_table(["Model", "Selection period", "Held-out period",
                                              "Held-out 80% / 95% coverage"], sel_rows),
        "table_forecast_replay": md_table(["Input to the primary model", "Mean CRPS", "vs finalized",
                                           "80% coverage"], replay_rows),
        "table_forecast_latest": md_table(["Series", "Next week: median (80% interval)",
                                           "In 4 weeks: median (80% interval)", "Batch reporting"], latest_rows),
        "fc_primary": label[primary],
        "fc_selection_end": f"{fem.SELECTION_END:%Y}",
        "fc_test_rel": f"{bp.loc[('test', primary), 'relative_crps']:.2f}",
        "fc_test_gain": pct(1 - bp.loc[("test", primary), "relative_crps"]),
        "fc_test_cov80": pct(bp.loc[("test", primary), "coverage80"]),
        "fc_test_cov95": pct(bp.loc[("test", primary), "coverage95"]),
        "fc_n_series_test": str(len(tp)),
        "fc_n_beat": str(int((tp["relative_crps"] < 1).sum())),
        "fc_losers": join_words([f"{series_name(r.iso3, r.disease)} ({r.relative_crps:.2f})"
                                 for r in losers.itertuples()]),
        "fc_worst_model": label[worst_model],
        "fc_worst_range": num_range(worst, lambda v: f"{v:.1f}"),
        "fc_ens_test": f"{ens_test:.2f}",
        "fc_n_forecasts": num(len(fc["backtest"]) // len(fcm.MODELS)),
        "fc_train_weeks": str(fcm.TRAIN_WEEKS),
        "fc_min_history": str(fcm.MIN_HISTORY),
        "fc_refit": str(fcm.REFIT_EVERY),
        "fc_naive_window": str(fcm.NAIVE_WINDOW),
        "fc_cap": f"{fcm.CAP_FACTOR:g}",
        "fc_ridge": f"{fcm.RIDGE:g}",
        "fc_n_origins_replay": str(fc["replay"]["origin"].nunique()),
        "fc_realtime_penalty": f"{mean_crps['as_reported'] / mean_crps['finalized']:.2f}",
        "fc_nowcast_gain": pct(1 - mean_crps["nowcast_adjusted"] / mean_crps["as_reported"]),
        "fc_nowcast_helps": fmt_change(helps),
        "fc_nowcast_hurts": fmt_change(hurts),
        "fc_origin": f"{origin:%d %b %Y}",
        "fc_no_forecast": join_words([series_name(*k) for k in no_forecast]),
        "fc_inactive_years": f"{(nowcast.TRAIN_WEEKS + nowcast.D_MAX) / 52:.0f}",
        "fc_material_crps": f"{MATERIAL_CRPS:g}",
    }


def _population_values(ctx: dict) -> dict:
    pop = ctx["population"]
    elig = ctx["audit"][ctx["audit"]["Forecast eligibility"].str.startswith("Eligible")]
    keys = set(zip(elig["ISO3"], elig["Disease"]))
    p = pop[[k in keys for k in zip(pop["iso3"], pop["disease"])]].sort_values("pct_official_mean")
    w = p.iloc[0]
    kind = "poultry" if w["disease"] == "HPAI" else "pig"
    return {"lowest_official_pop": f"only {w['pct_official_mean']:.0f}% of {COUNTRIES[w['iso3']]}'s {kind} values "
                                   "are official",
            "population_lag_years": str(config.POPULATION_LAG_YEARS),
            "population_last_anchor": f"July {int(pop['last_year'].max()) + config.POPULATION_LAG_YEARS}"}


ITEM_LABEL = {"chickens": "Chickens", "swine_pigs": "Pigs"}
CENSUS_CLOSE = 0.10  # census within 10% of FAOSTAT counts as agreeing


def _robustness_values(ctx: dict) -> dict:
    """GLW census vs FAOSTAT, population offset sensitivity and the WAHIS independence check."""
    cc = ctx["census_check"].dropna(subset=["ratio"])
    far = cc[(cc["ratio"] - 1).abs() > CENSUS_CLOSE].sort_values("ratio")
    census_rows = [[COUNTRIES[r.iso3], ITEM_LABEL[r.item], str(r.census_year), num(r.census_head / 1e6, 1),
                    num(r.faostat_head / 1e6, 1) + (" (imputed)" if r.faostat_flag == "I" else ""),
                    f"{r.ratio:.2f}"] for r in far.itertuples()]
    off = ctx["detection"]["offset"]
    perf = off["summary"].set_index("offset")
    w, wo = perf.loc["with offset"], perf.loc["without offset"]
    bys = off["by_series"]
    ind = ctx["detection"]["independence"].set_index(["method", "window_sources"])
    other = ind.xs("other sources only", level="window_sources")
    woah = ind.xs("WOAH-sourced", level="window_sources")
    return {
        "table_census": md_table(["Country", "Animals", "Census year", "Census (million)",
                                  "FAOSTAT, same year (million)", "Census / FAOSTAT"], census_rows),
        "census_n": str(len(cc)),
        "census_close_n": str(len(cc) - len(far)),
        "census_close_pct": f"{100 * CENSUS_CLOSE:.0f}%",
        "offset_n_series": str(len(bys)),
        "offset_alarms_with": num(bys["alarms_with"].sum()),
        "offset_weeks_differing": num(bys["weeks_differing"].sum()),
        "offset_wahis_with": pct(w["wahis_event_sensitivity"]),
        "offset_wahis_without": pct(wo["wahis_event_sensitivity"]),
        "offset_sim_with": pct(w["sim_sensitivity_k5"]),
        "offset_sim_without": pct(wo["sim_sensitivity_k5"]),
        "indep_n_other": num(other["events"].iloc[0]),
        "indep_n_woah": num(woah["events"].iloc[0]),
        "indep_other_hit": pct_range(other["sensitivity"]),
        "indep_other_chance": pct_range(other["chance_hit_rate"]),
        "indep_woah_hit": pct_range(woah["sensitivity"]),
        "indep_woah_chance": pct_range(woah["chance_hit_rate"]),
    }


def _priority_values(ctx: dict) -> dict:
    from . import priority as prm

    pr, check, wb, std = ctx["priority"], ctx["priority_check"], ctx["wild_birds"], ctx["standardized"]
    alerts = pr[pr["priority"] != "Low"]
    rows = [[series_name(r.iso3, r.disease), r.priority, r.confidence, r.reasons,
             r.notes.replace("; ", ". ") or "—"] for r in alerts.itertuples()]
    low = pr[pr["priority"] == "Low"]
    counts = pr["priority"].value_counts()
    parts = [f"{counts.get(p, 0)} {p}" for p in ("High", "Medium", "Low")]
    high = [series_name(r.iso3, r.disease) for r in pr[pr["priority"] == "High"].itertuples()]
    if len(check):
        event_rows = [[f"{r.event_start_date:%d %b %Y}", series_name(r.iso3, r.disease), r.reason, r.priority]
                      for r in check.itertuples()]
        missed = sorted({series_name(r.iso3, r.disease)
                         for r in check[check["priority"].isin(["Low", "not monitored"])].itertuples()})
        verdict = (f"The alerts would have missed the events in {join_words(missed)}: "
                   f"{'its' if len(missed) == 1 else 'their'} recent counts stay within the usual range, so no "
                   "signal fires." if missed else "Every one of them is in a series with at least one signal.")
        wahis = (f"{len(check)} WAHIS events started in the last 26 weeks:\n\n"
                 + md_table(["Event start", "Series", "Reason", "Priority now"], event_rows) + "\n\n" + verdict)
    else:
        wahis = "No WAHIS event started in the last 26 weeks."
    birds = std[(std["exclude_reason"] == "non-poultry bird") & (std["disease"] == "HPAI")]
    by_country = wb.groupby("iso3").size().sort_values(ascending=False)
    return {
        "table_priority": md_table(["Series", "Priority", "Confidence", "Why", "Notes"], rows),
        "pr_summary": ", ".join(parts[:-1]) + f" and {parts[-1]} series"
                      + (f"; High: {join_words(high)}" if high else ""),
        "pr_low_note": (f"The other {len(low)} monitored series are Low: "
                        f"{join_words([series_name(r.iso3, r.disease) for r in low.itertuples()])}." if len(low) else ""),
        "pr_wahis_check": wahis,
        "pr_recent_weeks": str(prm.RECENT_WEEKS),
        "pr_backlog_min": str(prm.BACKLOG_MIN),
        "pr_rise_factor": f"{prm.RISE_FACTOR:g}",
        "pr_rise_min": str(prm.RISE_MIN),
        "pr_wild_weeks": str(prm.WILD_BIRD_WEEKS),
        "wb_n_records": num(len(birds)),
        "wb_n_other": num(len(birds) - wb["records"].sum()),
        "wb_n_events": num(len(wb)),
        "wb_countries": join_words([f"{COUNTRIES[c]} {n}" for c, n in by_country.items()]),
    }


API_CHECK_DATE = "24 Oct 2026"  # when the sources' programmatic access was last checked by hand (§10.2)


CLIMATE_MATERIAL = 0.02  # a held-out CRPS ratio within 2% of 1 counts as "no material difference"


def _climate_values(ctx: dict) -> dict:
    from . import covariates as cov
    from . import forecast_eval as fem

    fc, clim, socio = ctx["forecast"], ctx["climate"], ctx["socio"]
    cs = fc["climate_summary"].set_index("period")
    label = {"selection": f"Selection (before {fem.SELECTION_END:%Y})", "test": f"Held out ({fem.SELECTION_END:%Y} on)"}
    rows = [[label[p], str(int(x["series"])), num(x["forecasts"]), f"{x['nb_ar_relative_crps']:.2f}",
             f"{x['nb_ar_climate_relative_crps']:.2f}", f"{x['climate_vs_nb_ar']:.3f}",
             f"{int(x['series_climate_better'])} of {int(x['series'])}"] for p, x in cs.iterrows()]
    r = cs.loc["test", "climate_vs_nb_ar"]
    per = fc["climate_by_series"]
    per = per[per["period"] == "test"].sort_values("climate_vs_nb_ar")
    best, worst = per.iloc[0], per.iloc[-1]
    change = lambda x: f"{series_name(x.iso3, x.disease)} ({signed(100 * (x.climate_vs_nb_ar - 1), 0)}%)"  # noqa: E731
    if r < 1 - CLIMATE_MATERIAL:
        verdict = f"**Climate helps on held-out forecasts:** it lowers NB-AR's CRPS by {pct(1 - r)} (geometric mean over series)"
        future = ("National climate averages improved the forecasts modestly (§8.5); province-level climate "
                  "could add more, together with the province-level detection (§6.5).")
        summary = f"lower NB-AR's held-out forecast error by {pct(1 - r)}"
    elif r > 1 + CLIMATE_MATERIAL:
        verdict = f"**Climate makes held-out forecasts worse:** NB-AR's CRPS rises by {pct(r - 1)} (geometric mean over series)"
        future = ("National climate averages did not improve the forecasts (§8.5); province-level climate "
                  "might, together with the province-level detection (§6.5).")
        summary = f"make NB-AR's held-out forecasts worse (CRPS +{pct(r - 1)})"
    else:
        verdict = (f"**Climate makes no material difference on held-out forecasts:** the CRPS ratio is {r:.3f} "
                   f"(within {pct(CLIMATE_MATERIAL)} of 1)")
        future = ("National climate averages did not improve the forecasts (§8.5); province-level climate "
                  "might, together with the province-level detection (§6.5).")
        summary = "make no material difference to the forecasts"
    finding = (f"{verdict}. It is better in {int(cs.loc['test', 'series_climate_better'])} of "
               f"{int(cs.loc['test', 'series'])} series; the largest change either way is {change(best)} and "
               f"{change(worst)}."
               + ("" if r < 1 - CLIMATE_MATERIAL else " Recent weather adds little to a model that already knows "
                  "the recent counts and the season."))
    lo, hi = cov.CLIMATOLOGY_YEARS
    return {
        "table_climate": md_table(["Forecasts", "Series", "Forecasts per model", "NB-AR (rel. CRPS)",
                                   "NB-AR + climate (rel. CRPS)", "Climate / no climate", "Climate better in"], rows),
        "climate_finding": finding,
        "climate_future_note": future,
        "climate_summary": summary,
        "climate_ref_years": f"{lo}–{hi}",
        "climate_window": str(cov.ANOMALY_WINDOW),
        "climate_years": f"{clim['week'].min() + pd.Timedelta(days=6):%Y}–{clim['week'].max():%Y}",  # first week may start in Dec
        "wb_years": f"{int(socio['year'].min())}–{int(socio['year'].max())}",
    }


def _extension_values(ctx: dict) -> dict:
    std = ctx["standardized"]
    return {
        "n_media_records": num(((std["reporting_source"] == "Media") & std["include"]).sum()),
        "api_check_date": API_CHECK_DATE,
    }


def build_values(ctx: dict) -> dict[str, str]:
    values: dict[str, str] = {}
    for section in (_data_values, _scope_values, _dedup_values, _coverage_values, _delay_values,
                    _detection_values, _realtime_values, _subnational_values, _nowcast_values, _forecast_values,
                    _population_values, _robustness_values, _priority_values,
                    _extension_values, _climate_values):
        values.update(section(ctx))
    return values


GENERATED_NOTICE = ("<!-- GENERATED FILE: rendered from docs/report_template.md by `python run_pipeline.py`. "
                    "Edit the template, not this file. -->\n")


def write(ctx: dict, template: Path = TEMPLATE, output: Path = OUTPUT) -> Path:
    text = template.read_text(encoding="utf-8")
    text = re.sub(r"\A<!--.*?-->\s*", "", text, flags=re.S)  # drop the template's own header comment
    output.write_text(GENERATED_NOTICE + render(text, build_values(ctx)), encoding="utf-8", newline="\n")
    return output
