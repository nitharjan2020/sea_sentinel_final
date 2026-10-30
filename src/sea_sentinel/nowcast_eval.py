"""Retrospective evaluation of the nowcast on reconstructed as-of snapshots.

Every 4 weeks between ``FIRST_ORIGIN`` and the last week with a complete 52-week follow-up, the data
are cut back to what had been reported by that week (using report dates), the nowcast is run, and it
is scored against the count eventually reported within 52 weeks. The comparator is the naive
estimate: the reported count taken at face value.

The snapshots are reconstructed from report dates in the current extract. Later edits or deletions
of records cannot be seen, so this is a replay of reporting timing, not of the exact historical
database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .nowcast import D_MAX, crps, nowcast, series_completeness, within_target

FIRST_ORIGIN = pd.Timestamp("2023-01-02")
ORIGIN_STEP_WEEKS = 4
HORIZON_BINS = [-1, 3, 12, 26, D_MAX - 1]
HORIZON_LABELS = ["0-3", "4-12", "13-26", "27-51"]


def truth(outbreaks: pd.DataFrame) -> pd.Series:
    """Final count per (iso3, disease, week) reported within D_MAX weeks."""
    return within_target(outbreaks).groupby(["iso3", "disease", "event_week"]).size().rename("truth")


def origins(last_week: pd.Timestamp) -> pd.DatetimeIndex:
    end = last_week - pd.Timedelta(weeks=D_MAX)
    return pd.date_range(FIRST_ORIGIN, end, freq=f"{ORIGIN_STEP_WEEKS}W-MON")


def backtest(outbreaks: pd.DataFrame, series: list[tuple[str, str]], last_week: pd.Timestamp) -> pd.DataFrame:
    """One row per series x origin x weeks_back, with nowcast, naive estimate and truth."""
    frames = [nowcast(outbreaks, t, series) for t in origins(last_week)]
    bt = pd.concat(frames, ignore_index=True)
    tr = truth(outbreaks)
    idx = pd.MultiIndex.from_frame(bt[["iso3", "disease", "week"]])
    bt["truth"] = tr.reindex(idx).fillna(0).to_numpy()
    bt["err_nowcast"] = bt["q500"] - bt["truth"]
    bt["err_naive"] = bt["reported"] - bt["truth"]
    bt["crps_nowcast"] = crps(bt["reported"].to_numpy(), bt["post_size"].to_numpy(), bt["post_prob"].to_numpy(),
                              bt["truth"].to_numpy())
    bt["in80"] = (bt["q100"] <= bt["truth"]) & (bt["truth"] <= bt["q900"])
    bt["in95"] = (bt["q025"] <= bt["truth"]) & (bt["truth"] <= bt["q975"])
    bt["horizon"] = pd.cut(bt["weeks_back"], HORIZON_BINS, labels=HORIZON_LABELS)
    return bt


def _metrics(g: pd.DataFrame) -> pd.Series:
    mae_now, mae_naive = g["err_nowcast"].abs().mean(), g["err_naive"].abs().mean()
    return pd.Series({
        "n": len(g),
        "mean_truth": g["truth"].mean(),
        "mae_naive": mae_naive,
        "mae_nowcast": mae_now,
        "crps_naive": mae_naive,  # CRPS of a point estimate = absolute error
        "crps_nowcast": g["crps_nowcast"].mean(),
        "crps_skill_pct": 100 * (1 - g["crps_nowcast"].mean() / mae_naive) if mae_naive > 0 else np.nan,
        "bias_naive": g["err_naive"].mean(),
        "bias_nowcast": (g["nowcast_mean"] - g["truth"]).mean(),
        "coverage80": g["in80"].mean(),
        "coverage95": g["in95"].mean(),
    })


def summarize(bt: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return bt.groupby(by, observed=True).apply(_metrics, include_groups=False).reset_index().round(3)


def run(outbreaks: pd.DataFrame, audit: pd.DataFrame, last_week: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Latest nowcast plus the retrospective backtest, for series with at least detection coverage."""
    tiers = audit["Forecast eligibility"].str.match("Eligible|Conditional|Detection")
    series = list(audit.loc[tiers, ["ISO3", "Disease"]].itertuples(index=False, name=None))
    latest = nowcast(outbreaks, last_week, series)
    F = series_completeness(within_target(outbreaks), last_week)
    comp = pd.DataFrame(
        [{"iso3": k[0], "disease": k[1], "weeks_back": d, "completeness": f}
         for k, v in F.items() if len(k) == 2 and k in set(series) for d, f in enumerate(v)]
    )
    bt = backtest(outbreaks, series, last_week)
    return {
        "latest": latest,
        "completeness": comp,
        "backtest": bt,
        "summary": summarize(bt, ["batch_reporting", "disease", "horizon"]),
        "by_series": summarize(bt[bt["weeks_back"] <= 12], ["iso3", "disease", "batch_reporting"]),
    }
