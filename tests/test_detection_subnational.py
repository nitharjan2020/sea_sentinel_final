import numpy as np
import pandas as pd

from sea_sentinel import detection_subnational as ds
from sea_sentinel.detection import GUARD_WEEKS, run_detection


def unit_records(counts_by_unit: dict[str, np.ndarray], start="2015-01-05", iso3="VNM", disease="HPAI"):
    """One record per outbreak, per unit and week; also returns the national weekly panel."""
    n = len(next(iter(counts_by_unit.values())))
    weeks = pd.date_range(start, periods=n, freq="W-MON")
    rows = [(iso3, disease, unit, w) for unit, c in counts_by_unit.items()
            for w, k in zip(weeks, c) for _ in range(int(k))]
    o = pd.DataFrame(rows, columns=["iso3", "disease", "admin1", "event_week"])
    total = sum(counts_by_unit.values()).astype(float)
    panel = pd.DataFrame({"iso3": iso3, "disease": disease, "week": weeks, "outbreaks": total,
                          "coverage_status": "active", "log_population": np.nan})
    return o, panel


def test_units_are_monitored_only_after_enough_past_outbreaks_and_follow_national_window():
    c = np.zeros(60)
    c[10:35] = 1  # one outbreak per week: the 20th falls in week 29
    o, panel = unit_records({"A": c})
    panel.loc[:4, "outbreaks"] = np.nan  # before the national reporting window
    u = ds.unit_panel(o, panel)
    assert u["outbreaks"].iloc[:5].isna().all() and u["outbreaks"].iloc[5] == 0
    first = int(np.flatnonzero(u["monitored"])[0])
    assert first == 29 + 1 + GUARD_WEEKS  # 20 outbreaks in weeks before t - guard
    assert u["monitored"].iloc[first:].all()


def test_series_with_few_large_units_are_not_selected():
    big = {f"U{i}": np.full(30, 1) for i in range(5)}
    o, _ = unit_records(big)
    o_small, _ = unit_records({**big, "U4": np.full(30, 0) + (np.arange(30) < 10)})  # U4: 10 outbreaks
    assert ds.select_series(o, [("VNM", "HPAI")]) == [("VNM", "HPAI")]
    assert ds.select_series(o_small, [("VNM", "HPAI")]) == []


def test_scores_floor_empty_baselines_and_rank_bigger_clusters_higher():
    b = {"base_mean": np.zeros(3), "base_sd": np.zeros(3), "nb_mu": np.zeros(3), "nb_alpha": np.zeros(3),
         "evaluable": np.ones(3, dtype=bool)}
    for method in ("rolling", "negbin"):
        s = ds.scores(np.array([0.0, 1.0, 3.0]), b, method)
        assert s[0] == -np.inf  # no reported outbreak, no score
        assert np.isfinite(s[1]) and s[2] > s[1]  # empty baseline still gives a finite, ordered score


def test_calibrate_matches_target_alarm_rate():
    rng = np.random.default_rng(0)
    results = {k: {"ev": np.ones(500, dtype=bool), "sub": {"rolling": {"week_score": rng.normal(size=500)}}}
               for k in ("a", "b")}
    cut = ds.calibrate(results, "rolling", 0.05)
    rate = np.mean([(r["sub"]["rolling"]["week_score"] > cut).mean() for r in results.values()])
    assert abs(rate - 0.05) < 0.003


def test_localized_cluster_is_seen_by_admin1_not_nationally():
    rng = np.random.default_rng(5)
    n = 260
    counts = {f"U{i}": rng.poisson(2.0, n) for i in range(8)}
    counts["U0"][230] += 6  # one province: 6 extra outbreaks in a single week
    o, panel = unit_records(counts)
    base = run_detection(panel, [("VNM", "HPAI")])
    events = pd.DataFrame(columns=["iso3", "disease", "start_week"])
    ub = ds.unit_baselines(ds.unit_panel(o, base))
    u0 = ub[ub["admin1"] == "U0"].reset_index(drop=True)
    assert ds.scores(u0["outbreaks"].to_numpy(), {c: u0[c].to_numpy() for c in
                     ["base_mean", "base_sd", "nb_mu", "nb_alpha", "evaluable"]}, "rolling").argmax() == 230
    assert not base.loc[230, "rolling_alarm"]  # diluted in the national count of ~16 per week
    out = ds.run(o, base, events)
    assert out["series"] == [("VNM", "HPAI")]
    assert set(out["summary"]["system"]) == {"national", "admin1"}
    assert out["summary"]["matched"].sum() == 2  # one matched cut-off per method


def test_latest_alerts_name_the_province_behind_a_recent_alarm():
    rng = np.random.default_rng(5)
    n = 260
    counts = {f"U{i}": rng.poisson(2.0, n) for i in range(8)}
    counts["U3"][n - 3] += 8  # recent cluster in one province
    o, panel = unit_records(counts)
    base = run_detection(panel, [("VNM", "HPAI")])
    out = ds.run(o, base, pd.DataFrame(columns=["iso3", "disease", "start_week"]))
    latest = out["latest"]
    hit = latest[(latest["method"] == "rolling") & (latest["week"] == panel["week"].iloc[n - 3])]
    assert len(hit) == 1 and hit["admin1"].iloc[0] == "U3"
    assert (latest["week"] >= panel["week"].iloc[n - 12]).all()  # only the last 12 weeks
    assert (latest["score"] > latest["cut_off"]).all()
