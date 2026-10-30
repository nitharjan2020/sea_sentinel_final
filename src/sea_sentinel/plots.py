"""Static report figures (PNG) for the coverage audit and data description."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.dates  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .config import COUNTRIES, DISEASES, FIG_DIR  # noqa: E402

# Reference data-viz palette (light mode).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#f0efec"
SERIES = {"HPAI": "#2a78d6", "ASF": "#eb6834"}
DISEASE_TICK = {"HPAI": "HPAI\n(poultry)", "ASF": "ASF\n(swine)"}
SEQ = {250: "#86b6ef", 400: "#3987e5", 550: "#1c5cab", 700: "#0d366b"}

TIERS = [  # ordinal, weakest -> strongest
    ("Withhold: no dated records", "No records", NEUTRAL),
    ("Descriptive only (sparse)", "Descriptive", SEQ[250]),
    ("Detection / descriptive only", "Detection only", SEQ[400]),
    ("Eligible: forecast + detection", "Forecast-eligible", SEQ[700]),
]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "text.color": TEXT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    }
)


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def coverage_heatmap(audit: pd.DataFrame) -> None:
    """Country x disease grid: fill = eligibility tier, label = distinct outbreaks."""
    tier_idx = {full: i for i, (full, _, _) in enumerate(TIERS)}
    iso = list(COUNTRIES)
    dis = list(DISEASES)
    a = audit.set_index(["ISO3", "Disease"])
    grid = np.array([[tier_idx[a.loc[(c, d), "Forecast eligibility"]] for d in dis] for c in iso])
    counts = np.array([[a.loc[(c, d), "Distinct outbreaks"] for d in dis] for c in iso])

    fig, ax = plt.subplots(figsize=(4.4, 6.2))
    ax.imshow(grid, cmap=ListedColormap([t[2] for t in TIERS]), vmin=0, vmax=len(TIERS) - 1, aspect="auto")
    # 2px surface gaps between cells
    ax.set_xticks(np.arange(-0.5, len(dis)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(iso)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(len(iso)):
        for j in range(len(dis)):
            dark_fill = grid[i, j] >= 3
            ax.text(j, i, f"{counts[i, j]:,}" if counts[i, j] else "–", ha="center", va="center",
                    fontsize=9, color="#ffffff" if dark_fill else TEXT)

    ax.set_xticks(range(len(dis)), [DISEASE_TICK[d] for d in dis])
    ax.set_yticks(range(len(iso)), [f"{COUNTRIES[c]} ({c})" for c in iso])
    ax.xaxis.tick_top()
    ax.set_title("Distinct reported outbreaks and forecast eligibility\n", pad=8)
    ax.legend(
        handles=[Patch(facecolor=c, edgecolor=GRID, label=short) for _, short, c in TIERS][::-1],
        loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, title="Eligibility", alignment="left",
    )
    fig.text(0.01, -0.02, "Source: FAO EMPRES-i Plus (retrieved 2026-10-21), deduplicated, confirmed only.",
             color=TEXT_2, fontsize=7)
    _save(fig, "coverage_matrix.png")


def weekly_small_multiples(panel: pd.DataFrame, audit: pd.DataFrame, disease: str) -> None:
    """One panel per forecast-eligible country; shaded weeks have no reports (missing, not zero)."""
    eligible = audit[(audit["Disease"] == disease) & audit["Forecast eligibility"].str.startswith("Eligible")]
    countries = eligible["ISO3"].tolist()
    if not countries:
        return
    fig, axes = plt.subplots(len(countries), 1, figsize=(8, 1.5 * len(countries) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    color = SERIES[disease]
    for ax, iso3 in zip(axes, countries):
        s = panel[(panel["iso3"] == iso3) & (panel["disease"] == disease)]
        ax.fill_between(s["week"], 0, s["outbreaks"], step="post", color=color, alpha=0.25, linewidth=0)
        ax.step(s["week"], s["outbreaks"], where="post", color=color, linewidth=1.0)
        missing = s["outbreaks"].isna().to_numpy()
        ymax = max(1, np.nanmax(s["outbreaks"].to_numpy()) if (~missing).any() else 1)
        ax.fill_between(s["week"], 0, ymax * 1.05, where=missing, step="post", color=NEUTRAL, linewidth=0, zorder=0)
        ax.set_ylim(0, ymax * 1.05)
        ax.set_title(f"{COUNTRIES[iso3]}", fontsize=9, pad=2)
        ax.grid(axis="x", visible=False)
        ax.margins(x=0)
    axes[-1].set_xlabel("Week of observation")
    fig.supylabel("Outbreaks per week", color=TEXT_2, fontsize=9)
    fig.suptitle(f"{DISEASES[disease]}: weekly reported outbreaks, forecast-eligible series",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.text(0.01, -0.01, "Grey = before first / after last report (treated as missing, not zero).",
             color=TEXT_2, fontsize=7)
    fig.tight_layout()
    _save(fig, f"weekly_{disease.lower()}.png")


LABEL_OFFSETS = {"HPAI": (6, -4, "left"), "ASF": (7, -12, "left")}


def delay_ecdf(outbreaks: pd.DataFrame) -> None:
    """Cumulative share of outbreaks reported within N days of observation, by disease."""
    ok = outbreaks[outbreaks["delay_quality"] == "ok"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for disease, color in SERIES.items():
        d = np.sort(ok.loc[ok["disease"] == disease, "report_delay_days"].astype(float).to_numpy())
        if not len(d):
            continue
        y = np.arange(1, len(d) + 1) / len(d)
        ax.step(d + 1, y, where="post", color=color, linewidth=2, label=f"{disease} (n={len(d):,})")
        med = np.median(d)
        dx, dy, ha = LABEL_OFFSETS[disease]
        ax.annotate(f"{disease}: median {med:.0f} d", xy=(med + 1, 0.5), xytext=(dx, dy), ha=ha,
                    textcoords="offset points", fontsize=8, color=TEXT)
        ax.plot(med + 1, 0.5, "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 8, 29, 91, 366], ["0", "1", "7", "28", "90", "365"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Days from observation to report")
    ax.set_ylabel("Share of outbreaks reported")
    ax.set_ylim(0, 1.01)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Reporting delay by disease (all countries, EMPRES-i)")
    _save(fig, "reporting_delay_ecdf.png")


def _lag_labels(prof: pd.DataFrame) -> list[str]:
    """Integer day bins (lo, hi] -> '0', '1', '2–3', '4–7', ..."""
    return [f"{hi:g}" if lo + 1 >= hi else f"{lo + 1:g}–{hi:g}" for lo, hi in zip(prof["lower"], prof["upper"])]


def _km_labels(prof: pd.DataFrame) -> list[str]:
    return [f"<{hi:g}" if lo <= 0 else f"{lo:g}–{hi:g}" for lo, hi in zip(prof["lower"], prof["upper"])]


def dedup_review_figure(lag: pd.DataFrame, dist: pd.DataFrame, max_km: float, window_days: int) -> None:
    """Cross- vs same-source pair ratio by time lag and by distance; bars inside the threshold in blue."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    panels = [
        (axes[0], lag, _lag_labels(lag),
         lag["upper"] <= window_days, f"By time lag (pairs within {max_km:g} km)", "Days between records",
         f"merged: ≤ {window_days} d"),
        (axes[1], dist, _km_labels(dist),
         dist["upper"] <= max_km, f"By distance (pairs within {window_days} d)", "Distance between records (km)",
         f"merged: ≤ {max_km:g} km"),
    ]
    for ax, prof, labels, inside, title, xlabel, note in panels:
        x = np.arange(len(prof))
        ax.bar(x, prof["cross_to_same_ratio"], width=0.8, color=np.where(inside, SEQ[400], "#c3c2b7"))
        ax.set_xticks(x, labels, rotation=0, fontsize=7.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", visible=False)
        ax.text(0.98, 0.96, f"Blue = {note}", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=TEXT)
    axes[0].set_ylabel("Cross- / same-source pairs")
    fig.suptitle("Duplicate signal: excess cross-source pairs at short lags and distances",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.text(0.01, -0.03, "Same-source pairs measure ordinary outbreak clustering; the ratio is elevated only where "
             "cross-source duplicates add to it. Grey bars: background (not merged).", color=TEXT_2, fontsize=7)
    fig.tight_layout()
    _save(fig, "dedup_review.png")


METHOD_COLOR = {"rolling": "#2a78d6", "negbin": "#eb6834"}
METHOD_LABEL = {"rolling": "Rolling (mean + z·SD)", "negbin": "Negative binomial (q)"}
EXAMPLE_SERIES = (("VNM", "ASF"), ("PHL", "ASF"), ("KHM", "HPAI"))


def detection_tradeoff(perf: pd.DataFrame, n_series: int) -> None:
    """Sensitivity vs alarm rate across thresholds: WAHIS reference events and simulated outbreaks."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
    panels = [
        (axes[0], "wahis_event_sensitivity", "WAHIS reference events hit (−2 to +4 wk)"),
        (axes[1], "sim_sensitivity_k5", "Simulated outbreaks detected (size 5×SD)"),
    ]
    for ax, col, title in panels:
        for method, color in METHOD_COLOR.items():
            m = perf[perf["method"] == method].sort_values("alarm_rate")
            x = 100 * m["alarm_rate"]
            ax.plot(x, 100 * m[col], "-o", color=color, linewidth=2, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=METHOD_LABEL[method])
            if col == "wahis_event_sensitivity":
                ax.plot(x, 100 * m["wahis_chance_hit_rate"], "--", color=color, linewidth=1, alpha=0.8)
            d = m[m["default"]].iloc[0]
            lvl = f"z = {d['level']:g}" if method == "rolling" else f"q = {d['level']:g}"
            ax.plot(100 * d["alarm_rate"], 100 * d[col], "o", color=color, markersize=11,
                    markerfacecolor="none", markeredgewidth=1.5)
            above = method == "rolling"
            ax.annotate(f"default {lvl}", (100 * d["alarm_rate"], 100 * d[col]), xytext=(-4, 12) if above else (8, -14),
                        ha="center" if above else "left", textcoords="offset points", fontsize=8, color=TEXT)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Alarm rate on real data (% of weeks; upper bound on false alerts)")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(100))
    axes[0].text(0.98, 0.04, "dashed = chance hit rate", transform=axes[0].transAxes, ha="right",
                 fontsize=7.5, color=TEXT_2)
    axes[0].set_ylabel("Sensitivity")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(f"Detection trade-off across thresholds (macro average over {n_series} series)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    _save(fig, "detection_tradeoff.png")


def detection_examples(base: pd.DataFrame, events: pd.DataFrame, years: int = 3) -> None:
    """Recent weeks of selected series: counts, both thresholds, alarms and WAHIS event starts."""
    series = [s for s in EXAMPLE_SERIES if ((base["iso3"] == s[0]) & (base["disease"] == s[1])).any()]
    fig, axes = plt.subplots(len(series), 1, figsize=(9, 2.2 * len(series) + 0.5))
    axes = np.atleast_1d(axes)
    start = base["week"].max() - pd.Timedelta(weeks=52 * years)
    for ax, (iso3, disease) in zip(axes, series):
        s = base[(base["iso3"] == iso3) & (base["disease"] == disease) & (base["week"] >= start)]
        ax.bar(s["week"], s["outbreaks"], width=6, color="#c3c2b7", linewidth=0, label="Reported outbreaks")
        thr_max = np.nanmax(s[["rolling_threshold", "negbin_threshold"]].where(s["evaluable"], axis=0).to_numpy())
        top = np.nanmax([np.nanmax(s["outbreaks"].to_numpy()), thr_max, 1])
        for i, (method, color) in enumerate(METHOD_COLOR.items()):
            thr = s[f"{method}_threshold"].where(s["evaluable"])
            ax.step(s["week"], thr, where="mid", color=color, linewidth=1.5, label=f"{METHOD_LABEL[method].split(' (')[0]} threshold")
            a = s[s[f"{method}_alarm"]]
            ax.plot(a["week"], np.full(len(a), top * (1.12 + 0.1 * i)), "v", color=color, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1, linestyle="none", label=f"{METHOD_LABEL[method].split(' (')[0]} alarm")
        ev = events[(events["iso3"] == iso3) & (events["disease"] == disease) & (events["start_week"] >= start)]
        for j, w in enumerate(ev["start_week"]):
            ax.axvline(w, color=TEXT, linewidth=0.8, linestyle=":", label="WAHIS event start" if j == 0 else None)
        ax.set_ylim(0, top * 1.35)
        ax.set_title(f"{COUNTRIES[iso3]} — {disease}", fontsize=9, pad=2)
        ax.grid(axis="x", visible=False)
        ax.margins(x=0.01)
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.supylabel("Outbreaks per week", color=TEXT_2, fontsize=9)
    fig.suptitle(f"Detection in practice: last {years} years, default thresholds",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    _save(fig, "detection_examples.png")


INPUT_COLOR = {"as_reported": "#8a8984", "nowcast_adjusted": "#2a78d6"}
INPUT_LABEL = {"as_reported": "As reported at the time", "nowcast_adjusted": "Nowcast-corrected"}


def detection_realtime(curve: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Share of finalized-data alarms already flagged in real time, by weeks after the alarm week."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, method in zip(axes, METHOD_COLOR):
        n = summary[(summary["disease"] == "all") & (summary["method"] == method)]["final_alarms"].iloc[0]
        for inp, color in INPUT_COLOR.items():
            c = curve[(curve["method"] == method) & (curve["input"] == inp)].sort_values("lag")
            ax.plot(c["lag"], 100 * c["share_flagged"], "-o", color=color, linewidth=2, markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=INPUT_LABEL[inp])
        ax.set_title(f"{METHOD_LABEL[method]}: {n} finalized alarms", fontsize=9)
        ax.set_xlabel("Weeks after the alarm week")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(100))
    axes[0].set_ylabel("Already flagged in real time")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Real-time replay: when do finalized-data alarms first appear?", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    _save(fig, "detection_realtime.png")


SYSTEM_COLOR = {"national": "#8a8984", "admin1": "#2a78d6"}
SYSTEM_LABEL = {"national": "National", "admin1": "Admin1 (any province)"}


def detection_subnational(summary: pd.DataFrame) -> None:
    """Sensitivity vs alarm rate for national and subnational detection, per method."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True)
    for ax, method in zip(axes, METHOD_COLOR):
        for system, color in SYSTEM_COLOR.items():
            m = summary[(summary["system"] == system) & (summary["method"] == method)]
            curve = m[~m["matched"]].sort_values("alarm_rate")
            ax.plot(100 * curve["alarm_rate"], 100 * curve["sim_cluster_5"], "-o", color=color, linewidth=2,
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=SYSTEM_LABEL[system])
            mark = m[m["default"] if system == "national" else m["matched"]].iloc[0]
            ax.plot(100 * mark["alarm_rate"], 100 * mark["sim_cluster_5"], "o", color=color, markersize=11,
                    markerfacecolor="none", markeredgewidth=1.5)
        ax.set_title(METHOD_LABEL[method], fontsize=9)
        ax.set_xlabel("Alarm rate on real data (% of weeks)")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(100))
    axes[0].set_ylabel("Simulated 5-outbreak cluster\nin one province detected")
    axes[0].text(0.98, 0.04, "circled: national default and\nadmin1 at the same alarm rate", transform=axes[0].transAxes,
                 ha="right", fontsize=7.5, color=TEXT_2)
    axes[1].legend(frameon=False, loc="lower right")
    n = summary["series"].max()
    fig.suptitle(f"Subnational vs national detection of localized clusters (macro average over {n} series)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    _save(fig, "detection_subnational.png")


def detection_figures(det: dict[str, pd.DataFrame]) -> None:
    n_series = len(det["baselines"][["iso3", "disease"]].drop_duplicates())
    detection_tradeoff(det["performance"], n_series)
    detection_examples(det["baselines"], det["events"])
    if "realtime" in det:
        detection_realtime(det["realtime"]["curve"], det["realtime"]["summary"])
    if "subnational" in det:
        detection_subnational(det["subnational"]["summary"])


NOWCAST_SERIES = (("VNM", "ASF"), ("PHL", "ASF"), ("LAO", "ASF"), ("MYS", "ASF"), ("THA", "ASF"),
                  ("KHM", "HPAI"), ("PHL", "HPAI"), ("VNM", "HPAI"), ("IDN", "HPAI"))


def nowcast_latest(latest: pd.DataFrame) -> None:
    """Reported vs nowcast (median, 80% and 95% intervals) for the last 52 weeks of key series."""
    series = [s for s in NOWCAST_SERIES if ((latest["iso3"] == s[0]) & (latest["disease"] == s[1])).any()]
    ncol = 3
    nrow = int(np.ceil(len(series) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.4 * nrow + 0.6))
    axes = np.atleast_1d(axes).ravel()
    color = SERIES["HPAI"]
    for ax, (iso3, disease) in zip(axes, series):
        s = latest[(latest["iso3"] == iso3) & (latest["disease"] == disease)].sort_values("week")
        ax.bar(s["week"], s["reported"], width=6, color="#c3c2b7", linewidth=0, label="Reported so far")
        ax.fill_between(s["week"], s["q025"], s["q975"], step="mid", color=color, alpha=0.15, linewidth=0,
                        label="95% interval")
        ax.fill_between(s["week"], s["q100"], s["q900"], step="mid", color=color, alpha=0.30, linewidth=0,
                        label="80% interval")
        ax.step(s["week"], s["q500"], where="mid", color=color, linewidth=1.5, label="Nowcast (median)")
        ax.set_title(f"{COUNTRIES[iso3]} — {disease}", fontsize=8.5, pad=2)
        if s["batch_reporting"].iloc[0]:
            ax.text(0.01, 0.97, "batch reporting:\nlow confidence", transform=ax.transAxes, ha="left", va="top",
                    fontsize=7, color=TEXT_2)
        ax.grid(axis="x", visible=False)
        ax.margins(x=0.01)
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[len(series):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=8)
    as_of = latest["as_of"].max()
    fig.suptitle(f"Nowcast of weekly outbreaks as of week {pd.Timestamp(as_of):%d %b %Y} (last 52 weeks)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.supylabel("Outbreaks per week", color=TEXT_2, fontsize=9)
    fig.tight_layout()
    _save(fig, "nowcast_latest.png")


def nowcast_backtest(bt: pd.DataFrame) -> None:
    """Backtest by weeks back (series without batch reporting): bias of naive vs nowcast; PI coverage."""
    b = bt[~bt["batch_reporting"]]
    g = b.groupby("weeks_back")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    ax.plot(g["err_naive"].mean().index, g["err_naive"].mean(), color="#8a8984", linewidth=2, label="Reported so far (naive)")
    nb = (b["nowcast_mean"] - b["truth"]).groupby(b["weeks_back"]).mean()
    ax.plot(nb.index, nb, color=SERIES["HPAI"], linewidth=2, label="Nowcast (mean)")
    ax.set_title("Bias (outbreaks per series-week)", fontsize=9)
    ax.set_xlabel("Weeks before the as-of date")
    ax.legend(frameon=False, loc="lower right")
    ax = axes[1]
    for col, nominal, color, label in (("in80", 0.8, SERIES["HPAI"], "80% interval"),
                                       ("in95", 0.95, SERIES["ASF"], "95% interval")):
        ax.plot(g[col].mean().index, 100 * g[col].mean(), color=color, linewidth=2, label=label)
        ax.axhline(100 * nominal, color=color, linewidth=0.8, linestyle="--")
    ax.set_ylim(50, 101)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(100))
    ax.set_title("Prediction-interval coverage (dashed = nominal)", fontsize=9)
    ax.set_xlabel("Weeks before the as-of date")
    ax.legend(frameon=False, loc="lower right")
    n_orig = bt["as_of"].nunique()
    fig.suptitle(f"Nowcast backtest: {n_orig} as-of snapshots reconstructed from report dates "
                 "(series without batch reporting)", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    _save(fig, "nowcast_backtest.png")


def nowcast_figures(nc: dict[str, pd.DataFrame]) -> None:
    nowcast_latest(nc["latest"])
    nowcast_backtest(nc["backtest"])


FORECAST_COLOR = {"rolling_nb": "#2a78d6", "nb_ar": "#eb6834", "ensemble": "#1baf7a", "seasonal_nb": "#eda100"}


def forecast_skill(fc: dict) -> None:
    """Left: relative CRPS vs the naive benchmark by horizon. Right: held-out skill of the primary model per series."""
    from .forecast import MODEL_LABEL

    summ, primary = fc["summary"], fc["primary"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    ax.axhline(1, color=TEXT_2, linewidth=1, linestyle="--")
    ax.text(0.85, 1.02, "naive benchmark = 1", va="bottom", fontsize=7.5, color=TEXT_2)
    for m, color in FORECAST_COLOR.items():
        s = summ[summ["model"] == m].sort_values("horizon")
        ax.plot(s["horizon"], s["relative_crps"], "-o", color=color, linewidth=2.5 if m == primary else 1.5,
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(MODEL_LABEL[m] + (" (primary)" if m == primary else ""), (4, s["relative_crps"].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=TEXT)
    ax.set_xticks(list(range(1, 5)), [f"{h} wk" for h in range(1, 5)])
    ax.set_xlim(0.8, 5.6)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([0.5, 0.75, 1, 1.5, 2, 3])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Relative CRPS (lower is better)")
    ax.set_title("All periods, 10 eligible series", fontsize=9)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    bs = fc["by_series_test"]
    b = bs[bs["model"] == primary].sort_values("relative_crps")
    labels = [f"{COUNTRIES[i]} {d}" for i, d in zip(b["iso3"], b["disease"])]
    colors = np.where(b["relative_crps"] < 1, FORECAST_COLOR.get(primary, SERIES["HPAI"]), "#c3c2b7")
    ax.barh(labels, b["relative_crps"], color=colors, height=0.7)
    ax.axvline(1, color=TEXT_2, linewidth=1, linestyle="--")
    for y, v in enumerate(b["relative_crps"]):
        ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=7.5, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Relative CRPS vs naive (lower is better)")
    ax.set_title(f"{MODEL_LABEL[primary]} on held-out forecasts (2022 onwards)", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.suptitle("Forecast skill against the naive benchmark (finalized data)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    _save(fig, "forecast_skill.png")


def forecast_latest(latest: pd.DataFrame, nowcast_latest: pd.DataFrame, primary: str,
                    history_weeks: int = 26) -> None:
    """Recent history (reported and nowcast) with the primary model's 1-4-week forecast fan."""
    from .forecast import MODEL_LABEL

    lt = latest[latest["model"] == primary]
    series = list(lt[["iso3", "disease"]].drop_duplicates().itertuples(index=False, name=None))
    ncol = 3
    nrow = int(np.ceil(len(series) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.4 * nrow + 0.6))
    axes = np.atleast_1d(axes).ravel()
    color = FORECAST_COLOR.get(primary, SERIES["HPAI"])
    for ax, (iso3, disease) in zip(axes, series):
        nc = nowcast_latest[(nowcast_latest["iso3"] == iso3) & (nowcast_latest["disease"] == disease)
                            & (nowcast_latest["weeks_back"] < history_weeks)].sort_values("week")
        ax.bar(nc["week"], nc["reported"], width=6, color="#c3c2b7", linewidth=0, label="Reported so far")
        ax.step(nc["week"], nc["nowcast_mean"], where="mid", color=TEXT_2, linewidth=1.2, label="Nowcast (mean)")
        f = lt[(lt["iso3"] == iso3) & (lt["disease"] == disease) & (lt["input"] == "nowcast_adjusted")]
        f = f.sort_values("target_week")
        ax.fill_between(f["target_week"], f["q025"], f["q975"], color=color, alpha=0.15, linewidth=0,
                        label="95% interval")
        ax.fill_between(f["target_week"], f["q100"], f["q900"], color=color, alpha=0.3, linewidth=0,
                        label="80% interval")
        ax.plot(f["target_week"], f["q500"], "-o", color=color, linewidth=1.8, markersize=4,
                label="Forecast median (nowcast input)")
        g = lt[(lt["iso3"] == iso3) & (lt["disease"] == disease) & (lt["input"] == "as_reported")].sort_values("target_week")
        ax.plot(g["target_week"], g["q500"], "--", color=color, linewidth=1.2, label="Forecast median (as reported)")
        ax.set_title(f"{COUNTRIES[iso3]} — {disease}", fontsize=8.5, pad=2)
        if f["batch_reporting"].any():
            ax.text(0.01, 0.97, "batch reporting:\nlow confidence", transform=ax.transAxes, ha="left", va="top",
                    fontsize=7, color=TEXT_2)
        ax.grid(axis="x", visible=False)
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[len(series):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07), fontsize=8)
    origin = pd.Timestamp(lt["origin"].max())
    fig.suptitle(f"{MODEL_LABEL[primary]} forecasts for the 4 weeks after {origin:%d %b %Y}",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.supylabel("Outbreaks per week", color=TEXT_2, fontsize=9)
    fig.tight_layout()
    _save(fig, "forecast_latest.png")


def forecast_figures(fc: dict, nowcast_latest: pd.DataFrame) -> None:
    forecast_skill(fc)
    forecast_latest(fc["latest"], nowcast_latest, fc["primary"])


def make_all(audit: pd.DataFrame, panel: pd.DataFrame, outbreaks: pd.DataFrame) -> None:
    coverage_heatmap(audit)
    for disease in ("HPAI", "ASF"):
        weekly_small_multiples(panel, audit, disease)
    delay_ecdf(outbreaks)
