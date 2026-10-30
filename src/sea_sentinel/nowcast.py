"""Nowcasting recent outbreak counts from observed reporting delays.

For an as-of week ``T`` (data reported up to the end of week T), the reported count for event week
``w = T - d`` is incomplete: only outbreaks with reporting delay <= d weeks are known. The target is
the count eventually reported within ``D_MAX`` weeks.

1. **Delay distribution (chain ladder).** From the reporting triangle of outbreaks reported by T,
   development factors ``f_d = sum C(w, d+1) / sum C(w, d)`` are computed over event weeks for which
   both cells are observable (most recent ``TRAIN_WEEKS`` such weeks). Completeness is
   ``F(d) = prod_{j >= d} 1 / f_j`` with ``F(D_MAX) = 1``. This uses only data known at T and
   corrects for right truncation. Sparse series use the disease-level pooled triangle.
2. **Posterior (Poisson-Gamma thinning).** ``C ~ Poisson(lambda * F(d))``, with an empirical-Bayes
   Gamma(a, b) prior on lambda taken from the reference weeks: the ``PRIOR_WEEKS`` most recent weeks
   that are at least half complete, excluding the target week. The prior mean is the delay-adjusted
   level ``m = sum C / sum F``. The prior variance ``v`` is the variance of the adjusted weekly
   counts ``C / F`` (at least ``m``), so a volatile series gets a weak prior. Then
   ``a = m^2 / v`` and ``b = m / v``. The unreported remainder is
   ``R | C ~ NegBin(size = a + C, prob = (b + F) / (b + 1))``; the nowcast is ``C + R``, with exact
   quantiles even when C = 0.

**Batch reporting.** The model assumes each outbreak's delay is independent. Where most records
arrive in a few bulk uploads, a recent week is either entirely in or entirely missing, and the
nowcast is unreliable. ``batch_concentration`` measures this (share of records arriving in the
busiest 10% of report weeks), and such series are flagged.

Counts follow the weekly panel's definition: rows without an observation date are dated by report
week, so their delay is 0 by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

D_MAX = 52           # weeks; target = count reported within one year
TRAIN_WEEKS = 104    # event weeks per development factor
MIN_TRIANGLE = 30    # outbreaks needed in a series' triangle, else pool by disease
PRIOR_WEEKS = 13
PRIOR_MIN_F = 0.5
PRIOR_MIN_RATE = 0.1  # floor on the prior mean (outbreaks/week) so a quiet series is never certain-zero
BATCH_FLAG = 0.5      # share of records in the busiest 10% of report weeks above which nowcasts are flagged
QUANTILES = (0.025, 0.10, 0.50, 0.90, 0.975)


def delay_weeks(outbreaks: pd.DataFrame) -> pd.Series:
    return ((outbreaks["report_week"] - outbreaks["event_week"]).dt.days // 7).astype(int)


def within_target(outbreaks: pd.DataFrame) -> pd.DataFrame:
    """Records reported within ``D_MAX`` weeks: the nowcast target (later ones are batch back-fills)."""
    return outbreaks[delay_weeks(outbreaks) <= D_MAX]


def reported_as_of(outbreaks: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Records known at the end of week ``as_of`` (reconstructed from report dates)."""
    return outbreaks[outbreaks["report_week"] <= as_of]


def triangle(known: pd.DataFrame, as_of: pd.Timestamp, n_weeks: int) -> np.ndarray:
    """Cumulative reporting triangle ``C[i, d]`` for event weeks ``as_of - i`` (i = 0..n_weeks-1).

    Cells with ``d > i`` are not yet observable and are NaN.
    """
    i = ((as_of - known["event_week"]).dt.days // 7).to_numpy()
    d = delay_weeks(known).to_numpy()  # callers pass within_target() records, so d <= D_MAX
    keep = (i >= 0) & (i < n_weeks)
    counts = np.zeros((n_weeks, D_MAX + 1))
    np.add.at(counts, (i[keep], d[keep]), 1)
    cum = counts.cumsum(axis=1)
    observable = np.arange(D_MAX + 1)[None, :] <= np.arange(n_weeks)[:, None]
    return np.where(observable, cum, np.nan)


def completeness(tri: np.ndarray) -> np.ndarray:
    """F(d) for d = 0..D_MAX from chain-ladder development factors."""
    f = np.ones(D_MAX)
    for d in range(D_MAX):
        rows = np.flatnonzero(~np.isnan(tri[:, d + 1]))[:TRAIN_WEEKS]  # most recent observable
        lo, hi = np.nansum(tri[rows, d]), np.nansum(tri[rows, d + 1])
        f[d] = hi / lo if lo > 0 else 1.0
    F = np.ones(D_MAX + 1)
    for d in range(D_MAX - 1, -1, -1):
        F[d] = F[d + 1] / max(f[d], 1.0)
    return F


def prior(C: np.ndarray, F: np.ndarray, target: int) -> tuple[float, float]:
    """Empirical-Bayes Gamma(a, b) prior on the weekly rate from the reference weeks (target excluded)."""
    ref = [d for d in range(len(C)) if F[d] >= PRIOR_MIN_F and d != target][:PRIOR_WEEKS]
    if not ref:
        m, v = PRIOR_MIN_RATE, PRIOR_MIN_RATE
    else:
        c, f = C[ref], F[ref]
        m = max(c.sum() / f.sum(), PRIOR_MIN_RATE)
        v = max(np.var(c / f, ddof=1) if len(ref) > 1 else m, m)
    return m * m / v, m / v


def posterior(c, F, a, b):
    """Frozen NegBin for the unreported remainder R given reported count c and completeness F
    (scalars, or arrays for several weeks at once)."""
    return stats.nbinom(np.maximum(a + c, 1e-9), (b + F) / (b + 1.0))


def crps(reported: np.ndarray, size: np.ndarray, prob: np.ndarray, y: np.ndarray, tail: float = 1e-6) -> np.ndarray:
    """CRPS of the nowcast ``reported + NegBin(size, prob)`` for outcomes ``y``.

    Uses ``sum_n (P(N <= n) - 1{y <= n})^2`` over the integer support. For a point forecast this
    equals the absolute error, so it compares directly with the naive estimate's MAE.
    """
    out = np.empty(len(y))
    for i, (c, s, p, obs) in enumerate(zip(reported, size, prob, y)):
        dist = stats.nbinom(s, p)
        upper = int(max(obs, c + dist.ppf(1 - tail))) + 1
        n = np.arange(upper + 1)
        cdf = dist.cdf(n - c)  # P(R <= n - c); zero for n < c
        out[i] = np.sum((cdf - (obs <= n)) ** 2)
    return out


def batch_concentration(known: pd.DataFrame) -> float:
    """Share of records that arrived in the busiest 10% of report weeks (0 if no records)."""
    per_week = known.groupby("report_week").size().sort_values(ascending=False)
    if per_week.empty:
        return 0.0
    top = max(1, int(np.ceil(0.1 * len(per_week))))
    return float(per_week.iloc[:top].sum() / per_week.sum())


def nowcast_series(known: pd.DataFrame, as_of: pd.Timestamp, F: np.ndarray, horizon: int = D_MAX) -> pd.DataFrame:
    """Nowcast for event weeks ``as_of - d``, d = 0..horizon-1, given known records of one series."""
    n = horizon + PRIOR_WEEKS + D_MAX
    i = ((as_of - known["event_week"]).dt.days // 7).to_numpy()
    C = np.bincount(i[(i >= 0) & (i < n)], minlength=n).astype(float)
    Fd = np.array([F[min(d, D_MAX)] for d in range(n)])
    a, b = np.array([prior(C, Fd, d) for d in range(horizon)]).T
    c, f = C[:horizon], Fd[:horizon]
    post = posterior(c, f, a, b)  # one frozen distribution for all weeks: much faster than one per week
    q = c + post.ppf(np.array(QUANTILES)[:, None])
    d = np.arange(horizon)
    return pd.DataFrame({
        "as_of": as_of, "week": as_of - pd.to_timedelta(d, unit="W"), "weeks_back": d,
        "post_size": post.args[0], "post_prob": post.args[1],
        "reported": c, "completeness": f, "prior_mean": a / b,
        "nowcast_mean": c + post.mean(),
        **{f"q{int(1000 * p):03d}": v for p, v in zip(QUANTILES, q)},
    })


def series_completeness(outbreaks: pd.DataFrame, as_of: pd.Timestamp) -> dict[tuple[str, str], np.ndarray]:
    """F(d) as of ``as_of`` for every series, with disease-level pooling for sparse series."""
    known = reported_as_of(within_target(outbreaks), as_of)
    n = TRAIN_WEEKS + D_MAX + 1
    out, pooled = {}, {}
    for disease, g in known.groupby("disease"):
        pooled[disease] = completeness(triangle(g, as_of, n))
    for (iso3, disease), g in known.groupby(["iso3", "disease"]):
        recent = g[g["event_week"] > as_of - pd.Timedelta(weeks=n)]
        out[(iso3, disease)] = (
            completeness(triangle(g, as_of, n)) if len(recent) >= MIN_TRIANGLE else pooled[disease]
        )
        out[(iso3, disease, "pooled")] = len(recent) < MIN_TRIANGLE
    return out


def nowcast(outbreaks: pd.DataFrame, as_of: pd.Timestamp, series: list[tuple[str, str]],
            horizon: int = D_MAX) -> pd.DataFrame:
    """Nowcasts for all ``series`` as of week ``as_of`` (no information after that week is used)."""
    outbreaks = within_target(outbreaks)
    Fs = series_completeness(outbreaks, as_of)
    known = reported_as_of(outbreaks, as_of)
    frames = []
    for iso3, disease in series:
        if (iso3, disease) not in Fs:
            continue
        g = known[(known["iso3"] == iso3) & (known["disease"] == disease)]
        if not (g["event_week"] > as_of - pd.Timedelta(weeks=TRAIN_WEEKS + D_MAX)).any():
            continue  # inactive series: nothing reported in the window, no basis for a nowcast
        nc = nowcast_series(g, as_of, Fs[(iso3, disease)], horizon)
        recent = g[g["event_week"] > as_of - pd.Timedelta(weeks=TRAIN_WEEKS + D_MAX)]
        nc["batch_concentration"] = batch_concentration(recent)
        nc["batch_reporting"] = nc["batch_concentration"] > BATCH_FLAG
        nc.insert(0, "disease", disease)
        nc.insert(0, "iso3", iso3)
        nc["delay_model"] = "pooled" if Fs[(iso3, disease, "pooled")] else "series"
        frames.append(nc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
