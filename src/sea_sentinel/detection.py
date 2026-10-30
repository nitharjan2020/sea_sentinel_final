"""Prospective anomaly detection on the weekly outbreak panel.

Two baselines, each computed for week *t* from data strictly before ``t - guard`` (no look-ahead; the
guard band also keeps an emerging outbreak out of its own baseline):

* **Rolling** (EARS-C2 style): mean and SD of the previous ``window`` weeks. Alarm when
  ``y > mean + z * sd``.
* **Negative binomial**: GLM on the previous ``window`` weeks with annual harmonics and, where
  available, the log at-risk population as offset. Over-dispersion is estimated by moments
  (NB2: Var = mu + alpha * mu^2). Alarm when ``y`` exceeds the ``q`` quantile of NB(mu_t, alpha).

Missing weeks (outside a series' reporting window) are dropped from baselines, never treated as zero.
An alarm also requires at least one reported outbreak. Baselines are stored, so any threshold can be
applied afterwards without refitting.

Caveat: this runs on finalized data. In real time, the most recent weeks are under-reported
(see the reporting-delay summary); nowcasting addresses that separately.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

GUARD_WEEKS = 2
ROLLING_WINDOW, ROLLING_MIN_OBS = 52, 26
NB_WINDOW, NB_MIN_OBS = 104, 52
REWEIGHT_CUTOFF = 2.58  # Pearson residual above which a past week is down-weighted (Noufaily et al. 2013)

ROLLING_Z = (1.5, 2.0, 2.5, 3.0, 4.0)
NB_QUANTILES = (0.90, 0.95, 0.975, 0.99, 0.999)
DEFAULT_THRESHOLD = {"rolling": 3.0, "negbin": 0.99}


def rolling_baseline(y: pd.Series, window: int = ROLLING_WINDOW, guard: int = GUARD_WEEKS,
                     min_obs: int = ROLLING_MIN_OBS) -> pd.DataFrame:
    """Mean/SD of observed weeks in [t - guard - window, t - guard - 1]; NaN if too few observed."""
    past = y.shift(guard + 1).rolling(window, min_periods=min_obs)
    mean = past.mean()
    sd = past.std().where(mean.isna(), past.std().fillna(0.0))  # SD 0 only where a mean exists
    return pd.DataFrame({"base_mean": mean, "base_sd": sd})


def _design(week: pd.Series) -> np.ndarray:
    angle = 2 * np.pi * week.dt.day_of_year.to_numpy() / 365.25
    return np.column_stack([np.ones(len(week)), np.sin(angle), np.cos(angle)])


def _nb_alpha(y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
    """Weighted NB2 method-of-moments dispersion, floored at 0 (Poisson)."""
    den = np.sum(w * mu**2)
    return max(0.0, float(np.sum(w * ((y - mu) ** 2 - mu)) / den)) if den > 0 else 0.0


def _robust_weights(y: np.ndarray, mu: np.ndarray, n_params: int, cut: float = REWEIGHT_CUTOFF) -> np.ndarray:
    """Farrington-style down-weighting of past outbreak weeks: w = (cut / r)^2 where r > cut.

    Residuals are quasi-Poisson, r = (y - mu) / sqrt(phi * mu). The NB variance is not used here:
    its dispersion is inflated by the very epidemics being down-weighted, which would mask them.
    """
    mu = np.maximum(mu, 1e-12)
    phi = max(1.0, float(np.sum((y - mu) ** 2 / mu) / max(len(y) - n_params, 1)))
    r = (y - mu) / np.sqrt(phi * mu)
    with np.errstate(divide="ignore"):  # r = 0 only where the weight is 1 anyway
        return np.where(r > cut, (cut / r) ** 2, 1.0)


def nb_baseline(y: pd.Series, week: pd.Series, log_pop: pd.Series | None = None,
                window: int = NB_WINDOW, guard: int = GUARD_WEEKS, min_obs: int = NB_MIN_OBS,
                reweight: bool = True, targets=None) -> pd.DataFrame:
    """Predicted mean ``nb_mu`` and dispersion ``nb_alpha`` for each week from a past-only NB fit.

    With ``reweight`` the model is refitted once with past outbreak weeks down-weighted, so earlier
    epidemics in the window do not raise the expected level, and the dispersion is estimated from the
    weeks that were not flagged. ``targets`` (row positions) limits the fits to those weeks; the
    others stay NaN.
    """
    yv = y.to_numpy(float)
    X = _design(week)
    off = np.zeros(len(y)) if log_pop is None else log_pop.fillna(0.0).to_numpy(float)
    if log_pop is not None and log_pop.isna().all():
        off = np.zeros(len(y))
    mu = np.full(len(y), np.nan)
    alpha = np.full(len(y), np.nan)
    start = None
    for t in range(len(y)) if targets is None else targets:
        lo, hi = max(0, t - guard - window), t - guard  # baseline rows [lo, hi)
        if hi <= lo or np.isnan(yv[t]):
            continue
        obs = ~np.isnan(yv[lo:hi])
        if obs.sum() < min_obs:
            continue
        yb, Xb, ob = yv[lo:hi][obs], X[lo:hi][obs], off[lo:hi][obs]
        if yb.sum() == 0:
            mu[t], alpha[t] = 0.0, 0.0
            continue
        w = np.ones(len(yb))
        for _ in range(2 if reweight else 1):
            params = None
            if (yb > 0).sum() >= 3:  # enough signal for seasonal terms
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = sm.GLM(yb, Xb, family=sm.families.Poisson(), offset=ob,
                                     var_weights=w).fit(start_params=start)
                    if np.all(np.isfinite(res.params)) and np.abs(res.params[1:]).max() < 10:
                        params = start = res.params
                except Exception:  # noqa: BLE001 - fall back to the intercept-only model below
                    params = None
            if params is not None:
                mu_b = np.exp(Xb @ params + ob)
                mu_t = float(np.exp(X[t] @ params + off[t]))
            else:  # intercept-only: weighted rate per population unit
                rate = np.sum(w * yb) / np.sum(w * np.exp(ob))
                mu_b = rate * np.exp(ob)
                mu_t = rate * np.exp(off[t])
            # Dispersion from unflagged weeks only: even down-weighted epidemic weeks would dominate it.
            a = _nb_alpha(yb, mu_b, (w >= 1).astype(float))
            w = _robust_weights(yb, mu_b, Xb.shape[1] if params is not None else 1)
        mu[t], alpha[t] = mu_t, a
    return pd.DataFrame({"nb_mu": mu, "nb_alpha": alpha}, index=y.index)


def nb_upper(mu: np.ndarray, alpha: np.ndarray, q: float) -> np.ndarray:
    """q-quantile of NB2(mu, alpha); Poisson when alpha == 0. NaN where the baseline is missing."""
    mu = np.asarray(mu, float)
    alpha = np.asarray(alpha, float)
    out = np.full(mu.shape, np.nan)
    ok = ~np.isnan(mu) & ~np.isnan(alpha)
    pois = ok & (alpha < 1e-8)
    nb = ok & ~pois
    out[pois] = stats.poisson.ppf(q, mu[pois])
    n = 1.0 / alpha[nb]
    out[nb] = stats.nbinom.ppf(q, n, n / (n + mu[nb]))
    return out


def threshold(baselines: pd.DataFrame, method: str, level: float) -> np.ndarray:
    if method == "rolling":
        return (baselines["base_mean"] + level * baselines["base_sd"]).to_numpy(float)
    if method == "negbin":
        return nb_upper(baselines["nb_mu"].to_numpy(), baselines["nb_alpha"].to_numpy(), level)
    raise ValueError(method)


def alarms(y: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Alarm where the count exceeds the threshold and at least one outbreak was reported."""
    y = np.asarray(y, float)
    with np.errstate(invalid="ignore"):
        return (y > thr) & (y >= 1) & ~np.isnan(thr)


def run_detection(panel: pd.DataFrame, series: list[tuple[str, str]]) -> pd.DataFrame:
    """Baselines for every week of the given (iso3, disease) series, both methods side by side."""
    frames = []
    for iso3, disease in series:
        s = panel[(panel["iso3"] == iso3) & (panel["disease"] == disease)].sort_values("week").reset_index(drop=True)
        y = s["outbreaks"]
        log_pop = s["log_population"] if "log_population" in s else None
        base = pd.concat([rolling_baseline(y), nb_baseline(y, s["week"], log_pop)], axis=1)
        frames.append(pd.concat([s[["iso3", "disease", "week", "outbreaks", "coverage_status"]], base], axis=1))
    out = pd.concat(frames, ignore_index=True)
    # Weeks both methods can judge: common evaluation set for a fair comparison.
    out["evaluable"] = out["outbreaks"].notna() & out["base_mean"].notna() & out["nb_mu"].notna()
    for method, level in DEFAULT_THRESHOLD.items():
        thr = threshold(out, method, level)
        out[f"{method}_threshold"] = thr
        out[f"{method}_alarm"] = alarms(out["outbreaks"].to_numpy(), thr) & out["evaluable"]
    return out
