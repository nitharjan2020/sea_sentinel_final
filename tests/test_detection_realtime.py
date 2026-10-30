import numpy as np
import pandas as pd

from sea_sentinel import detection_realtime as dr
from sea_sentinel.detection import nb_baseline, run_detection


def records(counts, delays, start="2019-01-07", iso3="VNM", disease="ASF"):
    """One record per outbreak; ``delays`` (weeks) is a constant or a per-week array."""
    weeks = pd.date_range(start, periods=len(counts), freq="W-MON")
    delays = np.broadcast_to(delays, len(counts))
    rows = [(iso3, disease, w, w + pd.Timedelta(weeks=int(d)))
            for w, c, d in zip(weeks, counts, delays) for _ in range(int(c))]
    return pd.DataFrame(rows, columns=["iso3", "disease", "event_week", "report_week"]), weeks


def panel_of(counts, weeks, iso3="VNM", disease="ASF"):
    return pd.DataFrame({"iso3": iso3, "disease": disease, "week": weeks, "outbreaks": np.asarray(counts, float),
                         "coverage_status": "active", "log_population": np.nan})


def test_asof_counts_use_only_records_reported_by_then():
    counts = np.array([0, 0, 3, 2, 4, 1])
    o, weeks = records(counts, delays=[0, 0, 0, 1, 3, 0])
    pos = {w: i for i, w in enumerate(weeks)}
    ep = o["event_week"].map(pos).to_numpy()
    y = dr.asof_counts(ep, o["report_week"].to_numpy(), weeks[4], 4)
    # weeks 0-1 precede the first report (missing); week 3 reported one week late; week 4 not yet
    assert np.isnan(y[:2]).all() and list(y[2:]) == [3, 2, 0]


def test_nb_baseline_targets_match_full_fit():
    y = pd.Series(np.random.default_rng(3).poisson(4, 200).astype(float))
    w = pd.Series(pd.date_range("2020-01-06", periods=200, freq="W-MON"))
    full = nb_baseline(y, w)
    targets = np.array([120, 150, 199])
    part = nb_baseline(y, w, targets=targets)
    assert part.drop(index=targets).isna().all().all()
    np.testing.assert_allclose(part.loc[targets].to_numpy(), full.loc[targets].to_numpy(), rtol=1e-6)


def test_no_alarm_without_a_reported_outbreak():
    y = np.r_[np.random.default_rng(0).poisson(2, 150), 0].astype(float)
    week = pd.Series(pd.date_range("2020-01-06", periods=len(y), freq="W-MON"))
    inflated = y.copy()
    inflated[-1] = 50  # e.g. a nowcast correction of a week with nothing reported yet
    a = dr.judge(inflated, y, week, None)
    assert not a["rolling"][0] and not a["negbin"][0]
    reported = y.copy()
    reported[-1] = 1
    assert dr.judge(inflated, reported, week, None)["negbin"][0]


def test_without_reporting_delay_real_time_equals_finalized(monkeypatch):
    rng = np.random.default_rng(7)
    counts = rng.poisson(3, 300)
    counts[[215, 222, 230]] = 30  # spikes inside the replayed weeks (207-235)
    o, weeks = records(counts, delays=0)
    panel = panel_of(counts, weeks)
    last = weeks[-1]
    monkeypatch.setattr(dr, "REPLAY_START", last - pd.Timedelta(weeks=dr.D_MAX + 40))
    base = run_detection(panel, [("VNM", "ASF")])
    rp = dr.replay(o, panel, [("VNM", "ASF")], last)
    now = rp[rp["lag"] == 0].merge(base[["week", "rolling_alarm", "negbin_alarm"]], on="week")
    for m in ("rolling", "negbin"):
        g = now[now["method"] == m]
        assert (g["alarm"] == g[f"{m}_alarm"]).all()
    first = dr.first_alarms(rp, base)
    assert len(first) and (first["first_lag"] == 0).all()  # every finalized alarm is seen immediately
    assert dr.confirmed(rp, base)["final_alarm"].all()


def test_late_reports_delay_the_first_real_time_alarm(monkeypatch):
    rng = np.random.default_rng(11)
    counts = rng.poisson(3, 300)
    counts[215] = 30
    delays = np.zeros(300, dtype=int)
    delays[215] = 3  # the whole spike is reported three weeks late
    o, weeks = records(counts, delays)
    panel = panel_of(counts, weeks)
    last = weeks[-1]
    monkeypatch.setattr(dr, "REPLAY_START", last - pd.Timedelta(weeks=dr.D_MAX + 40))
    base = run_detection(panel, [("VNM", "ASF")])
    first = dr.first_alarms(dr.replay(o, panel, [("VNM", "ASF")], last), base)
    spike = first[(first["week"] == weeks[215]) & (first["input"] == "as_reported")]
    assert len(spike) and (spike["first_lag"] == 3).all()
