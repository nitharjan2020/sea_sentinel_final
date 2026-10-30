"""Probabilistic 1-4-week forecasts of weekly reported outbreaks.

Every model returns a full predictive distribution (a probability mass function over counts), so
all models are scored the same way (CRPS, log score, interval coverage).

* **Naive** (benchmark): median = last observed count; uncertainty from the empirical distribution of
  past h-week changes (last ``NAIVE_WINDOW`` weeks, symmetrised), as in the FluSight baseline.
* **Rolling NB**: negative binomial around the mean of the last 8 weeks; dispersion from the last
  26 weeks.
* **Seasonal NB**: negative binomial around the same calendar weeks (+/- 2) of up to 5 previous years.
* **NB-AR**: direct h-step count regression, refitted every ``REFIT_EVERY`` weeks on the last
  ``TRAIN_WEEKS`` weeks. Predictors: log(1 + y) of the last week, the 4-week mean and the 52-week
  mean, plus annual harmonics of the target week. Ridge-penalised Poisson GLM mean (stable on sparse
  windows), NB2 dispersion by moments, and the mean capped at ``CAP_FACTOR`` x the largest count seen
  so far (+5) so the model cannot extrapolate beyond anything in the history.
* **Ensemble**: equal-weight mixture of the four, fixed in advance (not tuned on the results).
* **NB-AR + climate** (test only, not in the ensemble or the model selection): NB-AR with the
  country's recent temperature and rainfall anomalies added as predictors (``covariates.py``). The
  anomalies at week T are known at T, so no future weather is needed. It runs only when a series is
  given climate data.

A forecast made at week T uses only weeks <= T. Missing weeks (outside a series' reporting window)
are ignored, never treated as zero.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HORIZONS = (1, 2, 3, 4)
MODELS = ("naive", "rolling_nb", "seasonal_nb", "nb_ar", "ensemble")
MODEL_LABEL = {"naive": "Naive (benchmark)", "rolling_nb": "Rolling NB", "seasonal_nb": "Seasonal NB",
               "nb_ar": "NB-AR", "ensemble": "Ensemble", "nb_ar_climate": "NB-AR + climate"}
BASE_MODELS = ("naive", "rolling_nb", "seasonal_nb", "nb_ar")
TRAIN_WEEKS = 156
MIN_HISTORY = 104       # observed weeks needed before the first forecast
REFIT_EVERY = 4
NAIVE_WINDOW = 104
MU_FLOOR = 0.01         # avoids a degenerate point mass at zero after long quiet spells
LOG_SCORE_FLOOR = -10.0
RIDGE = 0.05            # L2 penalty on standardised-scale NB-AR coefficients (intercept unpenalised)
CAP_FACTOR = 2.0
QUANTILES = (0.025, 0.10, 0.50, 0.90, 0.975)


# --------------------------------------------------------------------------- distributions
def nb_pmf(mu: float, alpha: float, K: int) -> np.ndarray:
    """NB2(mu, alpha) pmf on 0..K, renormalised; Poisson when alpha is ~0."""
    mu = max(mu, MU_FLOOR)
    k = np.arange(K + 1)
    if alpha < 1e-8:
        p = stats.poisson.pmf(k, mu)
    else:
        n = 1.0 / alpha
        p = stats.nbinom.pmf(k, n, n / (n + mu))
    return p / p.sum()


def nb_upper(mu: float, alpha: float, q: float = 0.9999) -> int:
    mu = max(mu, MU_FLOOR)
    if alpha < 1e-8:
        return int(stats.poisson.ppf(q, mu))
    n = 1.0 / alpha
    return int(stats.nbinom.ppf(q, n, n / (n + mu)))


def moments_alpha(values: np.ndarray) -> float:
    v = values[~np.isnan(values)]
    if len(v) < 3:
        return 0.0
    m = v.mean()
    return max(0.0, (v.var(ddof=1) - m) / m**2) if m > 0 else 0.0


def scores(pmf: np.ndarray, y: float) -> dict[str, float]:
    """CRPS, floored log score, median, mean and central-interval membership for outcome y."""
    cdf = np.cumsum(pmf)
    k = np.arange(len(pmf))
    yi = int(round(y))
    q = {p: int(np.searchsorted(cdf, p - 1e-12)) for p in QUANTILES}
    return {
        "crps": float(np.sum((cdf - (yi <= k)) ** 2)),
        "log_score": float(max(np.log(pmf[yi]) if pmf[yi] > 0 else -np.inf, LOG_SCORE_FLOOR)),
        "median": q[0.5], "mean": float(np.dot(k, pmf)),
        "abs_error": abs(q[0.5] - yi),
        "in80": q[0.10] <= yi <= q[0.90],
        "in95": q[0.025] <= yi <= q[0.975],
    }


def quantiles(pmf: np.ndarray) -> dict[str, int]:
    cdf = np.cumsum(pmf)
    return {f"q{int(1000 * p):03d}": int(np.searchsorted(cdf, p - 1e-12)) for p in QUANTILES}


# --------------------------------------------------------------------------- features
def features(y: pd.Series) -> np.ndarray:
    """Per week: log1p of last value, 4-week mean and 52-week mean (NaN where y is missing)."""
    m4 = y.rolling(4, min_periods=1).mean()
    m52 = y.rolling(52, min_periods=13).mean()
    F = np.column_stack([np.log1p(y), np.log1p(m4), np.log1p(m52)])
    F[y.isna().to_numpy()] = np.nan
    return F


def harmonics(weeks: pd.Series) -> np.ndarray:
    angle = 2 * np.pi * weeks.dt.day_of_year.to_numpy() / 365.25
    return np.column_stack([np.sin(angle), np.cos(angle)])


class Series:
    """One weekly series with the precomputed pieces every model needs."""

    def __init__(self, weeks: pd.Series, y: pd.Series, climate: np.ndarray | None = None):
        self.weeks = weeks.reset_index(drop=True)
        self.C = None if climate is None else np.asarray(climate, float)  # weeks x climate predictors
        self.y = y.reset_index(drop=True).astype(float)
        self.yv = self.y.to_numpy()
        self.F = features(self.y)
        self.H = harmonics(self.weeks)
        self._fits: dict[tuple[int, int, bool], tuple | None] = {}

    def harmonics_at(self, i: int) -> np.ndarray:
        """Harmonics for row i, extrapolating dates past the end of the series."""
        if i < len(self.weeks):
            return self.H[i]
        w = self.weeks.iloc[-1] + pd.Timedelta(weeks=i - len(self.weeks) + 1)
        a = 2 * np.pi * w.day_of_year / 365.25
        return np.array([np.sin(a), np.cos(a)])

    # ----------------------------------------------------------------------- NB-AR
    def _predictors(self, s, climate: bool) -> np.ndarray:
        """Origin-week predictors: AR features, plus the climate anomalies if requested."""
        return np.column_stack([self.F[s], self.C[s]]) if climate else self.F[s]

    def nb_ar_fit(self, R: int, h: int, climate: bool = False):
        """Direct h-step model fitted on pairs (s, s+h) with s + h <= R; cached per refit point."""
        key = (R, h, climate)
        if key not in self._fits:
            s = np.arange(max(0, R - h - TRAIN_WEEKS + 1), R - h + 1)
            ok = ~np.isnan(self._predictors(s, climate)).any(axis=1) & ~np.isnan(self.yv[s + h])
            s = s[ok]
            fit = None
            if len(s) >= 52 and (self.yv[s + h] > 0).sum() >= 3:
                X = np.column_stack([np.ones(len(s)), self._predictors(s, climate), self.H[s + h]])
                penalty = np.r_[0.0, np.full(X.shape[1] - 1, RIDGE)]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = sm.GLM(self.yv[s + h], X, family=sm.families.Poisson()).fit_regularized(
                            alpha=penalty, L1_wt=0.0)
                    beta = np.asarray(res.params)
                    if np.all(np.isfinite(beta)):
                        mu = np.exp(np.clip(X @ beta, -20, 15))
                        alpha = max(0.0, float(np.sum((self.yv[s + h] - mu) ** 2 - mu) / np.sum(mu**2)))
                        fit = (beta, alpha)
                except Exception:  # noqa: BLE001 - no fit: the caller falls back to rolling NB
                    fit = None
            self._fits[key] = fit
        return self._fits[key]

    # ----------------------------------------------------------------------- per-model parameters
    def params(self, T: int, h: int) -> dict[str, tuple]:
        """Distribution parameters of every base model for target T + h, using weeks <= T only."""
        y = self.yv
        last = y[T]
        m8 = np.nanmean(y[max(0, T - 7):T + 1])
        a26 = moments_alpha(y[max(0, T - 25):T + 1])
        out = {"rolling_nb": ("nb", m8, a26)}

        idx = [T + h - 52 * k + j for k in range(1, 6) for j in range(-2, 3)]
        vals = np.array([y[i] for i in idx if 0 <= i <= T])
        vals = vals[~np.isnan(vals)]
        out["seasonal_nb"] = ("nb", vals.mean(), moments_alpha(vals)) if len(vals) >= 10 else out["rolling_nb"]

        s = np.arange(max(0, T - NAIVE_WINDOW - h + 1), T - h + 1)
        d = y[s + h] - y[s]
        d = d[~np.isnan(d)]
        if len(d) >= 20:
            out["naive"] = ("empirical", np.clip(np.round(np.concatenate([last + d, last - d])), 0, None))
        else:
            out["naive"] = ("nb", last, 0.0)

        cap = CAP_FACTOR * np.nanmax(y[: T + 1]) + 5
        variants = (("nb_ar", False),) + ((("nb_ar_climate", True),) if self.C is not None else ())
        for name, climate in variants:
            fit = self.nb_ar_fit(T - T % REFIT_EVERY, h, climate)
            x_T = self._predictors(np.array([T]), climate)[0]
            if fit is not None and not np.isnan(x_T).any():
                beta, alpha = fit
                x = np.concatenate([[1.0], x_T, self.harmonics_at(T + h)])
                out[name] = ("nb", float(min(np.exp(np.clip(x @ beta, -20, 15)), cap)), alpha)
            else:  # NB-AR + climate falls back to NB-AR, which falls back to rolling NB
                out[name] = out["nb_ar"] if climate else out["rolling_nb"]
        return out


def _upper(p: tuple) -> int:
    return int(p[1].max()) if p[0] == "empirical" else nb_upper(p[1], p[2])


def _pmf(p: tuple, K: int) -> np.ndarray:
    if p[0] == "empirical":
        pmf = np.bincount(p[1].astype(int), minlength=K + 1)[: K + 1].astype(float)
        return pmf / pmf.sum()
    return nb_pmf(p[1], p[2], K)


def pmfs(params: dict[str, tuple], y_obs: float | None = None) -> dict[str, np.ndarray]:
    """Pmfs of all models (incl. the ensemble) on a common support 0..K."""
    K = max(_upper(p) for p in params.values())
    if y_obs is not None:
        K = max(K, int(round(y_obs)))
    K += 1
    out = {m: _pmf(params[m], K) for m in BASE_MODELS}
    out["ensemble"] = np.mean([out[m] for m in BASE_MODELS], axis=0)
    if "nb_ar_climate" in params:
        out["nb_ar_climate"] = _pmf(params["nb_ar_climate"], K)
    return out


def forecast_origin(series: Series, T: int, horizons=HORIZONS) -> list[dict]:
    """Forecast rows (model x horizon) made at row T, with quantiles; no outcome needed."""
    rows = []
    for h in horizons:
        P = pmfs(series.params(T, h))
        for m, pmf in P.items():
            rows.append({"horizon": h, "model": m, "mean": float(np.dot(np.arange(len(pmf)), pmf)), **quantiles(pmf)})
    return rows
