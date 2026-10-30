import numpy as np
import pandas as pd

from sea_sentinel import forecast as fc
from sea_sentinel import forecast_eval as fe


def make_series(values, start="2020-01-06"):
    weeks = pd.Series(pd.date_range(start, periods=len(values), freq="W-MON"))
    return fc.Series(weeks, pd.Series(values, dtype=float))


def background(n=260, rate=4, seed=3):
    return np.random.default_rng(seed).poisson(rate, n).astype(float)


def test_forecast_does_not_look_ahead():
    y = background()
    y2 = y.copy()
    y2[201:] = 400  # change everything after the origin
    T = 200
    for h in fc.HORIZONS:
        a, b = make_series(y).params(T, h), make_series(y2).params(T, h)
        for m in a:
            assert np.allclose(np.atleast_1d(a[m][1]), np.atleast_1d(b[m][1]))


def test_pmfs_are_distributions_and_ensemble_is_their_mean():
    P = fc.pmfs(make_series(background()).params(200, 2), y_obs=30)
    for p in P.values():
        assert np.isclose(p.sum(), 1) and (p >= 0).all()
    base = np.mean([P[m] for m in ("naive", "rolling_nb", "seasonal_nb", "nb_ar")], axis=0)
    assert np.allclose(P["ensemble"], base)


def test_scores_point_mass():
    pmf = np.zeros(10)
    pmf[3] = 1
    s = fc.scores(pmf, 5)
    assert np.isclose(s["crps"], 2) and s["abs_error"] == 2  # CRPS of a point forecast = |error|
    assert s["log_score"] == fc.LOG_SCORE_FLOOR and not s["in80"]
    assert fc.scores(pmf, 3)["in95"]


def test_nb_ar_cannot_explode_on_sparse_history():
    y = np.zeros(260)
    y[[30, 90, 150, 200]] = 1  # long quiet spell with a few isolated outbreaks
    y[250:258] = [2, 8, 4, 6, 7, 3, 5, 6]  # sudden activity
    ser = make_series(y)
    for h in fc.HORIZONS:
        kind, mu, alpha = ser.params(257, h)["nb_ar"]
        assert mu <= fc.CAP_FACTOR * np.nanmax(y[:258]) + 5


def test_missing_weeks_are_ignored_not_zero():
    y = background(rate=10)
    y[180:200] = np.nan
    m = make_series(y).params(205, 1)["rolling_nb"][1]
    assert m > 5  # mean over the observed weeks, not dragged down by the gap


def test_model_selection_uses_only_the_selection_period():
    rows = []
    for iso3 in ("A", "B"):
        for origin, good in ((pd.Timestamp("2020-06-01"), "rolling_nb"), (pd.Timestamp("2024-06-01"), "nb_ar")):
            for m in fc.MODELS:
                crps = 1.0 if m == "naive" else (0.5 if m == good else 0.9)
                rows.append({"iso3": iso3, "disease": "ASF", "origin": origin, "horizon": 1, "model": m,
                             "crps": crps, "log_score": -1.0, "abs_error": 1, "in80": True, "in95": True})
    primary, by_period = fe.select_model(pd.DataFrame(rows))
    assert primary == "rolling_nb"  # best before SELECTION_END, even though nb_ar wins afterwards
    t = by_period.set_index(["period", "model"])
    assert t.loc[("test", "nb_ar"), "relative_crps"] < t.loc[("test", "rolling_nb"), "relative_crps"]
