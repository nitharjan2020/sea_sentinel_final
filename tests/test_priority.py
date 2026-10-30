import numpy as np
import pandas as pd

from sea_sentinel import priority

END = pd.Timestamp("2026-10-12")


def std_rows(animal_types, event="2026-06-01", report="2026-06-10", iso3="PHL", reason="non-poultry bird"):
    n = len(animal_types)
    return pd.DataFrame({"record_id": range(n), "iso3": iso3, "disease": "HPAI", "exclude_reason": reason,
                         "animal_type": animal_types, "event_date": event, "report_date": report,
                         "admin1": [f"A{i}" for i in range(n)]})


def test_wild_bird_events_keep_wild_birds_only():
    s = std_rows(["Wild - Duck", "Wild - Pigeon", "Captive - Tiger (Panthera tigris)", "Wild - civet",
                  "Captive - Pheasant"])
    s = pd.concat([s, std_rows(["Wild - Duck"], reason=None)])  # in-scope rows are not wild-bird records
    ev = priority.wild_bird_events(s)
    assert sorted(ev["species"]) == ["Wild - Duck", "Wild - Pigeon"]  # "pigeon" is not a mammal


def test_wild_bird_signal_uses_only_reports_known_by_then():
    ev = priority.wild_bird_events(pd.concat([
        std_rows(["Wild - Duck"], event="2026-06-01", report="2026-06-10"),
        std_rows(["Wild - Duck"], event="2026-09-01", report="2026-11-01", iso3="VNM"),   # reported later
        std_rows(["Wild - Duck"], event="2025-01-01", report="2025-01-05", iso3="KHM"),   # too old
    ]))
    sig = priority.wild_bird_signal(ev, END)
    assert list(sig.index) == ["PHL"]


def inputs(det_weeks=(), sub_weeks=(), reported=4.0, nowcast=4.0, batch=False, forecast=1.0, past=1.0):
    """Minimal pipeline outputs for one series (VNM HPAI)."""
    audit = pd.DataFrame({"ISO3": ["VNM"], "Disease": ["HPAI"], "Forecast eligibility": ["Eligible: forecast + detection"]})
    weeks = pd.date_range(END - pd.Timedelta(weeks=99), END, freq="W-MON")
    panel = pd.DataFrame({"iso3": "VNM", "disease": "HPAI", "week": weeks, "outbreaks": past,
                          "population_head": 1e6})
    det = pd.DataFrame({"iso3": "VNM", "disease": "HPAI", "week": pd.to_datetime(list(det_weeks)),
                        "rolling_alarm": True, "negbin_alarm": False})
    sub = pd.DataFrame({"iso3": "VNM", "disease": "HPAI", "week": pd.to_datetime(list(sub_weeks)),
                        "method": "rolling", "admin1": "Nam Dinh"})
    nc = pd.DataFrame({"iso3": "VNM", "disease": "HPAI", "weeks_back": range(4), "reported": reported / 4,
                       "nowcast_mean": nowcast / 4, "batch_reporting": batch})
    fc = pd.DataFrame({"iso3": "VNM", "disease": "HPAI", "model": "rolling_nb", "input": "nowcast_adjusted",
                       "horizon": [1, 2, 3, 4], "mean": forecast})
    return audit, panel, det, sub, nc, fc


def build(**kw):
    audit, panel, det, sub, nc, fc = inputs(**kw)
    wb = priority.wild_bird_events(std_rows(["Wild - Duck"], iso3=kw.pop("wb_iso3", "KHM")))
    return priority.build(audit, panel, det, sub, nc, fc, "rolling_nb", wb, END).iloc[0]


def test_no_signal_is_low_and_each_signal_adds_a_point():
    assert build()["priority"] == "Low"
    r = build(det_weeks=["2026-09-07"])
    assert r["priority"] == "Medium" and r["points"] == 1 and "national alarm (rolling" in r["reasons"]
    r = build(det_weeks=["2026-09-07"], sub_weeks=["2026-09-07"], nowcast=10.0)
    assert r["priority"] == "High" and r["points"] == 3 and "Nam Dinh" in r["reasons"]


def test_old_alarms_and_small_changes_do_not_count():
    r = build(det_weeks=["2026-06-01"], nowcast=5.0, forecast=1.1)  # alarm > 12 weeks ago, backlog 1, forecast flat
    assert r["points"] == 0


def test_batch_reporting_turns_backlog_and_forecast_into_notes():
    r = build(nowcast=20.0, forecast=5.0, batch=True)
    assert r["points"] == 0 and r["confidence"] == "low"
    assert "not counted" in r["notes"] and not r["reporting_backlog"] and not r["rising_forecast"]
    r = build(nowcast=20.0, forecast=5.0)
    assert r["points"] == 2 and r["reporting_backlog"] and r["rising_forecast"]


def test_wahis_check_lists_recent_events_with_their_priority():
    audit, panel, det, sub, nc, fc = inputs()
    pr = priority.build(audit, panel, det, sub, nc, fc, "rolling_nb", priority.wild_bird_events(std_rows([])), END)
    wahis = pd.DataFrame({"iso3": ["VNM", "VNM", "THA"], "disease": "HPAI", "event_id": [1, 2, 3], "reason": "x",
                          "event_start_date": ["2026-09-01", "2020-01-01", "2026-08-01"]})
    out = priority.wahis_check(pr, wahis, END)
    assert list(out["event_id"]) == [1, 3] and list(out["priority"]) == ["Low", "not monitored"]
    assert np.isnan(out.loc[1, "points"])
