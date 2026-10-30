import numpy as np
import pandas as pd

from sea_sentinel import nowcast as nc


def synthetic(n_weeks=300, rate=10, delays=(0, 1, 2, 4), probs=(0.4, 0.3, 0.2, 0.1), seed=1,
              start="2020-01-06", iso3="VNM", disease="ASF"):
    """Outbreak records with a known weekly delay distribution."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(start, periods=n_weeks, freq="W-MON")
    rows = []
    for w in weeks:
        for _ in range(rng.poisson(rate)):
            d = rng.choice(delays, p=probs)
            rows.append((iso3, disease, w, w + pd.Timedelta(weeks=int(d))))
    return pd.DataFrame(rows, columns=["iso3", "disease", "event_week", "report_week"]), weeks


def test_chain_ladder_recovers_known_delay_distribution():
    df, weeks = synthetic()
    as_of = weeks[-1]
    F = nc.completeness(nc.triangle(nc.reported_as_of(df, as_of), as_of, nc.TRAIN_WEEKS + nc.D_MAX + 1))
    expected = np.cumsum([0.4, 0.3, 0.2, 0, 0.1])  # P(delay <= d) for d = 0..4
    assert np.allclose(F[:5], expected, atol=0.03)
    assert np.allclose(F[4:], 1.0)


def test_no_look_ahead():
    df, weeks = synthetic()
    as_of = weeks[200]
    a = nc.nowcast(df, as_of, [("VNM", "ASF")])
    later = df.copy()
    later.loc[later["report_week"] > as_of, "report_week"] += pd.Timedelta(weeks=3)  # change the future
    b = nc.nowcast(later, as_of, [("VNM", "ASF")])
    pd.testing.assert_frame_equal(a, b)


def test_nowcast_corrects_truncation_and_covers_truth():
    df, weeks = synthetic(rate=20)
    as_of = weeks[250]
    out = nc.nowcast(df, as_of, [("VNM", "ASF")], horizon=4)
    truth = df.groupby("event_week").size()
    out["truth"] = out["week"].map(truth).fillna(0)
    assert (out["reported"] < out["truth"]).iloc[0]                    # recent week is incomplete
    assert (out["q025"] <= out["truth"]).all() and (out["truth"] <= out["q975"]).all()
    assert abs(out["nowcast_mean"].iloc[0] - out["truth"].iloc[0]) < abs(out["reported"].iloc[0] - out["truth"].iloc[0])


def test_complete_week_is_returned_unchanged():
    post = nc.posterior(c=7, F=1.0, a=5.0, b=0.5)
    assert post.ppf(0.999) == 0  # nothing left to report


def test_zero_reported_with_low_completeness_relies_on_prior():
    a, b = nc.prior(np.array([0.0, 10, 12, 8, 11, 9]), np.array([0.05, 0.9, 0.95, 1, 1, 1]), target=0)
    post = nc.posterior(c=0, F=0.05, a=a, b=b)
    assert 5 < post.mean() < 15  # pulled towards the recent level of ~10/week, not 0


def test_crps_of_point_forecast_is_absolute_error():
    # completeness 1 -> remainder is a point mass at 0, so CRPS = |reported - truth|
    tiny = np.array([1e-9])
    assert np.isclose(nc.crps(np.array([5.0]), tiny, np.array([1.0]), np.array([8.0]))[0], 3.0)


def test_batch_concentration():
    steady = pd.DataFrame({"report_week": pd.date_range("2024-01-01", periods=50, freq="W-MON")})
    batchy = pd.DataFrame({"report_week": [pd.Timestamp("2024-06-03")] * 90
                           + list(pd.date_range("2024-01-01", periods=10, freq="W-MON"))})
    assert nc.batch_concentration(steady) < 0.2
    assert nc.batch_concentration(batchy) > nc.BATCH_FLAG
