"""End-to-end data pipeline: ingest -> standardise -> deduplicate -> weekly panel + population offsets
-> coverage audit -> anomaly detection (national + subnational) -> nowcasting -> real-time detection replay
-> forecasting (each with backtests) -> priority alerts -> figures -> technical report.

Run with ``python -m sea_sentinel.pipeline`` (or ``python run_pipeline.py``) from the project root.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from . import (coverage, covariates, denominators, detection_eval, detection_realtime, detection_subnational,
               forecast_eval, ingest, nowcast_eval, priority)
from .config import FIG_DIR, INTERIM_DIR, POPULATION_LAG_YEARS, PROCESSED_DIR, RAW_SOURCES, TABLE_DIR
from .dedup import deduplicate
from .load import load_empres, load_faostat, load_glw_census, load_wahis_events
from .panel import build_weekly_panel
from .standardize import (standardize_empres, standardize_faostat, standardize_glw_census, standardize_wahis,
                          week_start)

log = logging.getLogger("sea_sentinel")


def _raw(source_id: str, contains: str = "") -> list:
    return [ingest.raw_path(s.raw_name) for s in RAW_SOURCES if s.source_id == source_id and contains in s.raw_name]


def last_complete_week(extraction_date: pd.Timestamp) -> pd.Timestamp:
    """Monday of the last full Monday-Sunday week strictly before the extraction date."""
    this_week = week_start(pd.Series([extraction_date])).iloc[0]
    return this_week - pd.Timedelta(weeks=1)


def run(force_ingest: bool = False, make_figures: bool = True, review_dedup: bool = False) -> dict[str, pd.DataFrame]:
    for d in (INTERIM_DIR, PROCESSED_DIR, TABLE_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log.info("1/12 ingest raw files")
    manifest = ingest.ingest(force=force_ingest)

    log.info("2/12 load + standardise")
    empres_raw, empres_meta = load_empres(_raw("SRC003")[0])
    log.info("  EMPRES-i: %d rows (%d repaired), exported %s",
             len(empres_raw), empres_raw.attrs["n_repaired_rows"], empres_meta.get("Date downloaded"))
    std = standardize_empres(empres_raw)
    std.to_csv(INTERIM_DIR / "empres_standardized.csv", index=False)

    wahis = standardize_wahis(load_wahis_events(_raw("SRC001")))
    wahis.to_csv(PROCESSED_DIR / "wahis_events.csv", index=False)

    stocks = standardize_faostat(load_faostat(_raw("SRC004")[0]))
    census = standardize_glw_census(load_glw_census(_raw("SRC005")[0]))
    stocks.to_csv(PROCESSED_DIR / "faostat_livestock_stocks.csv", index=False)

    climate = covariates.climate_weekly(pd.read_csv(_raw("SRC011")[0]))
    climate.to_csv(PROCESSED_DIR / "climate_weekly.csv", index=False)
    socio = covariates.socioeconomic(pd.read_csv(_raw("SRC007")[0]))
    socio.to_csv(PROCESSED_DIR / "socioeconomic_country_year.csv", index=False)
    log.info("  covariates: climate %d country-weeks (%s to %s), World Bank %d country-years",
             len(climate), climate["week"].min().date(), climate["week"].max().date(), len(socio))

    exclusions = std["exclude_reason"].fillna("included").value_counts().rename_axis("reason").reset_index()
    exclusions.to_csv(TABLE_DIR / "empres_scope_filter.csv", index=False)
    log.info("  in scope: %d / %d", std["include"].sum(), len(std))

    log.info("3/12 deduplicate")
    outbreaks = deduplicate(std[std["include"]])
    outbreaks.to_csv(PROCESSED_DIR / "outbreaks_dedup.csv", index=False)
    log.info("  %d in-scope records -> %d distinct outbreaks", std["include"].sum(), len(outbreaks))

    log.info("4/12 weekly panel")
    extraction = pd.to_datetime(empres_meta.get("Date downloaded", ""), format="%d/%m/%Y %H:%M", errors="coerce")
    if pd.isna(extraction):
        extraction = outbreaks["report_date"].max()
    end_week = last_complete_week(extraction)
    panel = build_weekly_panel(outbreaks[outbreaks["event_week"] <= end_week], end_week)
    annual_pop = denominators.annual_population(stocks)
    panel = denominators.add_population(panel, annual_pop, lag_years=POPULATION_LAG_YEARS)
    panel.to_csv(PROCESSED_DIR / "weekly_panel.csv", index=False)
    pop_summary = denominators.population_summary(annual_pop)
    pop_summary.to_csv(TABLE_DIR / "population_denominators.csv", index=False)
    census_check = denominators.census_check(census, stocks)
    census_check.to_csv(TABLE_DIR / "population_census_check.csv", index=False)
    log.info("  %d rows, weeks through %s; population offset on %d rows",
             len(panel), end_week.date(), panel["population_head"].notna().sum())

    log.info("5/12 coverage audit + delays")
    audit = coverage.coverage_audit(std, outbreaks, panel, wahis)
    audit.to_csv(TABLE_DIR / "coverage_matrix.csv", index=False)
    delays = coverage.delay_summary(outbreaks)
    delays.to_csv(TABLE_DIR / "reporting_delay_summary.csv", index=False)

    from .inventory import write_inventory

    inv_path = write_inventory(audit, manifest)
    log.info("  inventory written to %s", inv_path)

    log.info("6/12 anomaly detection")
    in_panel = outbreaks[outbreaks["event_week"] <= end_week]
    det = detection_eval.run(panel, audit, wahis, in_panel)
    base_cols = ["iso3", "disease", "week", "outbreaks", "evaluable", "base_mean", "base_sd", "nb_mu", "nb_alpha",
                 "rolling_threshold", "rolling_alarm", "negbin_threshold", "negbin_alarm"]
    det["baselines"][base_cols].to_csv(PROCESSED_DIR / "detection_baselines.csv", index=False)
    det["performance"].to_csv(TABLE_DIR / "detection_performance.csv", index=False)
    det["by_series"].to_csv(TABLE_DIR / "detection_by_series.csv", index=False)
    det["events"].to_csv(TABLE_DIR / "detection_reference_events.csv", index=False)
    det["latest"].to_csv(TABLE_DIR / "detection_latest_alerts.csv", index=False)
    det["independence"].to_csv(TABLE_DIR / "detection_wahis_independence.csv", index=False)
    det["offset"]["summary"].to_csv(TABLE_DIR / "detection_offset_sensitivity.csv", index=False)
    det["offset"]["by_series"].to_csv(TABLE_DIR / "detection_offset_sensitivity_by_series.csv", index=False)
    d = det["performance"][det["performance"]["default"]].set_index("method")
    for m in d.index:
        log.info("  %-7s alarm rate %.1f%% (%.1f/series-yr) | WAHIS events hit %.0f%% (chance %.0f%%) | "
                 "simulated k=5 %.0f%%", m, 100 * d.loc[m, "alarm_rate"], d.loc[m, "alarms_per_year"],
                 100 * d.loc[m, "wahis_event_sensitivity"], 100 * d.loc[m, "wahis_chance_hit_rate"],
                 100 * d.loc[m, "sim_sensitivity_k5"])
    sub = detection_subnational.run(in_panel, det["baselines"], det["events"])
    sub["summary"].to_csv(TABLE_DIR / "detection_subnational.csv", index=False)
    sub["by_series"].to_csv(TABLE_DIR / "detection_subnational_by_series.csv", index=False)
    sub["units"].to_csv(TABLE_DIR / "detection_subnational_units.csv", index=False)
    det["subnational"] = sub
    s = sub["summary"].set_index(["system", "method"])
    for m in ("rolling", "negbin"):
        n, a = s[s["default"]].loc[("national", m)], s[s["matched"]].loc[("admin1", m)]
        log.info("  %-7s same alarm budget (%.1f%% of weeks), one-province cluster of 5: national %.0f%% | "
                 "admin1 %.0f%% (%d series)", m, 100 * n["alarm_rate"], 100 * n["sim_cluster_5"],
                 100 * a["sim_cluster_5"], len(sub["series"]))

    log.info("7/12 nowcasting")
    nc = nowcast_eval.run(in_panel, audit, end_week)
    nc_cols = ["iso3", "disease", "as_of", "week", "weeks_back", "reported", "completeness", "prior_mean",
               "nowcast_mean", "q025", "q100", "q500", "q900", "q975", "batch_concentration", "batch_reporting",
               "delay_model"]
    nc["latest"][nc_cols].to_csv(PROCESSED_DIR / "nowcast_latest.csv", index=False)
    nc["completeness"].to_csv(TABLE_DIR / "reporting_completeness_latest.csv", index=False)
    nc["summary"].to_csv(TABLE_DIR / "nowcast_backtest_summary.csv", index=False)
    nc["by_series"].to_csv(TABLE_DIR / "nowcast_backtest_by_series.csv", index=False)
    r = nc["summary"]
    r = r[(~r["batch_reporting"]) & (r["horizon"] == "0-3")].set_index("disease")
    for dis in r.index:
        log.info("  %-6s last 4 wk (no batch reporting): bias %+.2f -> %+.2f /wk | CRPS %+.0f%% | 80%%/95%% PI cover "
                 "%.0f%%/%.0f%%", dis, r.loc[dis, "bias_naive"], r.loc[dis, "bias_nowcast"],
                 r.loc[dis, "crps_skill_pct"], 100 * r.loc[dis, "coverage80"], 100 * r.loc[dis, "coverage95"])

    log.info("8/12 real-time detection replay")
    rt = detection_realtime.run(in_panel, panel, audit, det["baselines"], end_week)
    rt["summary"].to_csv(TABLE_DIR / "detection_realtime_summary.csv", index=False)
    rt["curve"].to_csv(TABLE_DIR / "detection_realtime_curve.csv", index=False)
    rt["first"].to_csv(TABLE_DIR / "detection_realtime_alarms.csv", index=False)
    det["realtime"] = rt
    r = rt["summary"][rt["summary"]["disease"] == "all"].set_index(["method", "input"])
    for (m, i), x in r.iterrows():
        log.info("  %-7s %-16s %d finalized alarms: %.0f%% flagged within 4 wk | %.0f%% of real-time alarms "
                 "confirmed", m, i, x["final_alarms"], 100 * x["flagged_within_4"], 100 * x["share_confirmed"])

    log.info("9/12 forecasting")
    fc = forecast_eval.run(panel, audit, nc, climate)
    fc["climate_summary"].to_csv(TABLE_DIR / "forecast_climate_test.csv", index=False)
    fc["climate_by_series"].to_csv(TABLE_DIR / "forecast_climate_test_by_series.csv", index=False)
    fc["summary"].to_csv(TABLE_DIR / "forecast_backtest_summary.csv", index=False)
    fc["by_period"].to_csv(TABLE_DIR / "forecast_model_selection.csv", index=False)
    fc["by_series"].to_csv(TABLE_DIR / "forecast_backtest_by_series.csv", index=False)
    fc["by_series_test"].to_csv(TABLE_DIR / "forecast_heldout_by_series.csv", index=False)
    fc["replay_summary"].to_csv(TABLE_DIR / "forecast_realtime_replay.csv", index=False)
    fc["latest"].to_csv(PROCESSED_DIR / "forecast_latest.csv", index=False)
    t = fc["by_period"].set_index(["period", "model"])
    log.info("  primary model (chosen on forecasts before %s): %s | held-out relative CRPS vs naive %.2f",
             forecast_eval.SELECTION_END.date(), fc["primary"], t.loc[("test", fc["primary"]), "relative_crps"])
    c = fc["climate_summary"].set_index("period").loc["test"]
    log.info("  climate test (held out): NB-AR relative CRPS %.2f -> %.2f with climate; climate better in %d of %d "
             "series", c["nb_ar_relative_crps"], c["nb_ar_climate_relative_crps"], c["series_climate_better"],
             c["series"])

    log.info("10/12 priority alerts")
    wild_birds = priority.wild_bird_events(std)
    wild_birds.to_csv(TABLE_DIR / "wild_bird_hpai_events.csv", index=False)
    alerts = priority.build(audit, panel, det["latest"], det["subnational"]["latest"], nc["latest"], fc["latest"],
                            fc["primary"], wild_birds, end_week)
    alerts.to_csv(TABLE_DIR / "priority_alerts.csv", index=False)
    det["subnational"]["latest"].to_csv(TABLE_DIR / "detection_subnational_latest_alerts.csv", index=False)
    pcheck = priority.wahis_check(alerts, wahis, end_week)
    pcheck.to_csv(TABLE_DIR / "priority_wahis_check.csv", index=False)
    log.info("  %s", " | ".join(f"{k}: {v}" for k, v in alerts["priority"].value_counts().items()))
    for r in alerts[alerts["priority"] != "Low"].itertuples():
        log.info("  %-6s %s %-4s (%s confidence): %s", r.priority, r.iso3, r.disease, r.confidence, r.reasons)

    if make_figures:
        log.info("11/12 figures")
        from . import plots

        plots.make_all(audit, panel, outbreaks)
        plots.detection_figures(det)
        plots.nowcast_figures(nc)
        plots.forecast_figures(fc, nc["latest"])

    if review_dedup:
        from . import dedup_review

        dedup_review.main()

    log.info("12/12 technical report")
    from . import report

    path = report.write({
        "standardized": std, "outbreaks": outbreaks, "manifest": manifest, "wahis": wahis, "stocks": stocks,
        "extraction": extraction, "end_week": end_week, "n_repaired": empres_raw.attrs["n_repaired_rows"],
        "audit": audit, "delays": delays, "detection": det, "nowcast": nc, "forecast": fc,
        "population": pop_summary, "census_check": census_check, "priority": alerts,
        "priority_check": pcheck, "wild_birds": wild_birds, "climate": climate, "socio": socio,
    })
    log.info("  rendered %s", path.relative_to(path.parents[1]))

    return {"standardized": std, "outbreaks": outbreaks, "panel": panel, "audit": audit, "detection": det,
            "nowcast": nc, "forecast": fc,
            "delays": delays, "wahis": wahis, "stocks": stocks, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-ingest", action="store_true", help="re-copy raw files from the source folder")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--review-dedup", action="store_true",
                        help="also write the dedup threshold review (tables + figure, ~1 min)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(force_ingest=args.force_ingest, make_figures=not args.no_figures, review_dedup=args.review_dedup)


if __name__ == "__main__":
    main()
