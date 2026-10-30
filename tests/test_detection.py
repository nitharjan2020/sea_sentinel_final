import numpy as np
import pandas as pd

from sea_sentinel import detection_eval as de
from sea_sentinel.detection import alarms, nb_baseline, nb_upper, rolling_baseline, run_detection


def weekly(values, start="2020-01-06"):
    weeks = pd.date_range(start, periods=len(values), freq="W-MON")
    return pd.Series(values, dtype=float), pd.Series(weeks)


def panel_from(values, iso3="VNM", disease="ASF"):
    y, weeks = weekly(values)
    return pd.DataFrame({"iso3": iso3, "disease": disease, "week": weeks, "outbreaks": y,
                         "coverage_status": "active", "log_population": np.nan})


def background(n, seed=0):
    return np.random.default_rng(seed).poisson(3, n).astype(float)


def test_baselines_do_not_look_ahead():
    y = background(200)
    y2 = y.copy()
    y2[150:] = 500  # change the "future"
    w = weekly(y)[1]
    for fn in (lambda s: rolling_baseline(s), lambda s: nb_baseline(s, w)):
        a, b = fn(pd.Series(y)), fn(pd.Series(y2))
        # weeks up to 150 + guard(2) only use data before week 150
        pd.testing.assert_frame_equal(a.iloc[:153], b.iloc[:153])


def test_missing_weeks_are_not_zeros():
    y = background(120)
    y_gap = y.copy()
    y_gap[40:60] = np.nan
    base = rolling_baseline(pd.Series(y_gap))
    # week 100's window (47-97) holds 38 observed weeks; the gap must not drag the mean towards zero
    assert base["base_mean"].iloc[100] > 2
    zero_filled = rolling_baseline(pd.Series(np.nan_to_num(y_gap)))
    assert base["base_mean"].iloc[100] > zero_filled["base_mean"].iloc[100]
    # too few observed weeks -> no baseline at all (not a zero baseline)
    y_long_gap = y.copy()
    y_long_gap[40:80] = np.nan
    b = rolling_baseline(pd.Series(y_long_gap)).iloc[90]
    assert np.isnan(b["base_mean"]) and np.isnan(b["base_sd"])


def test_clear_spike_alarms_and_quiet_week_does_not():
    y = background(160)
    y[150] = 40
    det = run_detection(panel_from(y), [("VNM", "ASF")])
    assert det.loc[150, "rolling_alarm"] and det.loc[150, "negbin_alarm"]
    assert not det.loc[151, "negbin_alarm"]


def test_alarm_needs_at_least_one_outbreak():
    assert not alarms(np.array([0.0]), np.array([-1.0]))[0]
    assert alarms(np.array([1.0]), np.array([0.0]))[0]


def test_nb_upper_poisson_limit_and_monotone():
    assert nb_upper(np.array([2.0]), np.array([0.0]), 0.99)[0] == 6  # Poisson(2) 99th pct
    lo, hi = nb_upper(np.array([2.0, 2.0]), np.array([0.0, 1.0]), 0.99)
    assert hi > lo


def test_reweighting_limits_influence_of_past_epidemic():
    y = background(200)
    y[100:106] = 60  # past epidemic inside the baseline window
    w = weekly(y)[1]
    plain = nb_baseline(pd.Series(y), w, reweight=False).iloc[150]
    robust = nb_baseline(pd.Series(y), w, reweight=True).iloc[150]
    assert robust["nb_mu"] < plain["nb_mu"]
    assert robust["nb_alpha"] < plain["nb_alpha"]


def test_reference_event_hit_window_and_chance():
    y = background(200)
    y[150] = 40
    det = run_detection(panel_from(y), [("VNM", "ASF")])
    week150 = det.loc[150, "week"]
    wahis = pd.DataFrame({"iso3": ["VNM", "VNM"], "disease": "ASF", "event_id": ["1", "2"], "subtype": None,
                          "reason": "x", "event_start_date": [week150 + pd.Timedelta(weeks=1),   # alarm 1 wk before
                                                              week150 + pd.Timedelta(weeks=20)]})  # nothing nearby
    ev = de.reference_events(det, wahis)
    ref = de.evaluate_reference(det, ev)
    r = ref[(ref["method"] == "rolling") & (ref["level"] == 3.0)].iloc[0]
    assert r["n_events"] == 2 and r["events_detected"] == 1
    assert r["median_lag_weeks"] == -1
    assert r["chance_hit_rate"] < 0.5


def test_constant_population_level_does_not_change_nb_alarms():
    """The NB baseline refits its level, so scaling the population by a constant changes nothing."""
    y = background(220)
    y[[150, 180, 200]] = 25
    a, b = panel_from(y), panel_from(y)
    a["log_population"] = np.log(1e6)
    b["log_population"] = np.log(3.7e6)
    da, db = run_detection(a, [("VNM", "ASF")]), run_detection(b, [("VNM", "ASF")])
    assert (da["negbin_alarm"] == db["negbin_alarm"]).all() and da["negbin_alarm"].sum() >= 2
    np.testing.assert_allclose(da["nb_mu"], db["nb_mu"], rtol=1e-5)


def test_independence_splits_events_by_woah_sourced_records():
    y = background(200)
    y[150] = 40
    y[100] = 40
    det = run_detection(panel_from(y), [("VNM", "ASF")])
    w100, w150 = det.loc[100, "week"], det.loc[150, "week"]
    wahis = pd.DataFrame({"iso3": "VNM", "disease": "ASF", "event_id": ["1", "2", "3"], "subtype": None,
                          "reason": "x", "event_start_date": [w100, w150, w150 + pd.Timedelta(weeks=30)]})
    events = de.reference_events(det, wahis)
    outbreaks = pd.DataFrame({"iso3": "VNM", "disease": "ASF", "event_week": [w100, w150],
                              "reporting_sources": ["WOAH (former OIE)", "National authorities"]})
    src = de.event_sources(events, outbreaks)
    assert list(src) == ["WOAH-sourced", "other sources only", "no records"]
    ind = de.independence(det, events, outbreaks).set_index(["method", "window_sources"])
    assert ind.loc[("rolling", "other sources only"), "detected"] == 1
    assert ind.loc[("rolling", "no records"), "detected"] == 0
