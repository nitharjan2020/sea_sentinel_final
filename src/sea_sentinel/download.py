"""Download the open covariate sources into the source folder (``config.SOURCE_DIR``).

Run with ``python -m sea_sentinel.download``. The pipeline itself never uses the network: it copies
these files into ``data/raw/`` like every other download (``ingest.py``).

* **World Bank World Development Indicators** (SRC007): population by sex, land area, population
  density, GDP, GDP per capita and trade, per country and year, from the open API.
* **Natural Earth admin-0 boundaries** (SRC012): the 11 countries' outlines (1:50m), used only to
  average the climate grids per country.
* **NOAA CPC daily climate** (SRC011): the CPC Global Unified gauge-based daily precipitation and
  daily maximum / minimum temperature (0.5 degree), requested from NOAA PSL for a box around
  Southeast Asia, one year at a time. Each grid is averaged at download time over the land cells of
  each country (area-weighted; a country without a cell centre inside it, such as Singapore, uses the
  nearest cell), so the raw snapshot is a small country x day table instead of about 1 GB of grids.
  Values for Singapore (1 cell) and Brunei (2 cells) are therefore rough: Singapore's rainfall comes
  out at more than twice its station average. Neither country has a forecast-eligible series.
  Years before the current one are cached as per-year CSVs and not fetched again.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COUNTRIES, SOURCE_DIR

log = logging.getLogger("sea_sentinel")

WB_URL = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}?format=json&date={years}&per_page=2000"
WB_INDICATORS = {
    "SP.POP.TOTL.FE.IN": "population_female",
    "SP.POP.TOTL.MA.IN": "population_male",
    "AG.LND.TOTL.K2": "land_area_km2",
    "EN.POP.DNST": "population_density_km2",
    "NY.GDP.MKTP.CD": "gdp_usd",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "NE.TRD.GNFS.ZS": "trade_pct_gdp",
    "NE.EXP.GNFS.CD": "exports_usd",
    "NE.IMP.GNFS.CD": "imports_usd",
}
WB_YEARS = "2000:2025"
WB_FILE = "worldbank_wdi_sea.csv"

NE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"
NE_FILE = "naturalearth_admin0_sea.geojson"

CPC_URL = "https://psl.noaa.gov/thredds/ncss/grid/Datasets/{path}.{year}.nc"
CPC_VARS = {"precip": "cpc_global_precip/precip", "tmax": "cpc_global_temp/tmax", "tmin": "cpc_global_temp/tmin"}
CPC_BOX = {"north": 29, "south": -12, "west": 91, "east": 142}
CPC_FIRST_YEAR = 2003  # one year before the first outbreak series begins
CPC_FILE = "cpc_climate_sea_country_daily.csv"
CPC_CACHE = "cpc_cache"


def _get(url: str, retries: int = 3, timeout: int = 600) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sea-sentinel"}),
                                        timeout=timeout) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - retry, then re-raise
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- World Bank
def world_bank(out_dir: Path) -> Path:
    """Long table: iso3, year, indicator, variable, value."""
    rows = []
    for code, name in WB_INDICATORS.items():
        url = WB_URL.format(countries=";".join(COUNTRIES), indicator=code, years=WB_YEARS)
        meta, data = json.loads(_get(url))
        for d in data or []:
            rows.append({"iso3": d["countryiso3code"], "year": int(d["date"]), "indicator": code, "variable": name,
                         "value": d["value"], "source_updated": meta.get("lastupdated")})
    path = out_dir / WB_FILE
    pd.DataFrame(rows).sort_values(["iso3", "indicator", "year"]).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- boundaries
def boundaries(out_dir: Path) -> Path:
    """Natural Earth admin-0 features of the 11 countries, keyed by ISO3."""
    world = json.loads(_get(NE_URL))
    keep = []
    for f in world["features"]:
        p = f["properties"]
        iso3 = next((p.get(k) for k in ("ISO_A3", "ISO_A3_EH", "ADM0_A3") if p.get(k) in COUNTRIES), None)
        if iso3:
            keep.append({"type": "Feature", "properties": {"iso3": iso3, "name": p.get("NAME")},
                         "geometry": f["geometry"]})
    missing = set(COUNTRIES) - {f["properties"]["iso3"] for f in keep}
    if missing:
        raise RuntimeError(f"Natural Earth file lacks {sorted(missing)}")
    path = out_dir / NE_FILE
    path.write_text(json.dumps({"type": "FeatureCollection", "source": NE_URL, "features": keep}), encoding="utf-8")
    return path


def country_weights(geojson: Path, lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    """Per country, area weights (cos latitude) over grid cells whose centre lies inside it.

    A country with no cell centre inside (too small for a 0.5 degree grid) gets its nearest cell.
    """
    from matplotlib.path import Path as MplPath

    glon, glat = np.meshgrid(lon, lat)
    pts = np.column_stack([glon.ravel(), glat.ravel()])
    area = np.cos(np.deg2rad(glat))
    out = {}
    for f in json.loads(geojson.read_text(encoding="utf-8"))["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        inside = np.zeros(len(pts), dtype=bool)
        verts = []
        for poly in polys:
            ring = np.asarray(poly[0])  # exterior ring; lakes inside countries are ignored
            inside |= MplPath(ring).contains_points(pts)
            verts.append(ring)
        w = np.where(inside.reshape(glat.shape), area, 0.0)
        if not w.any():
            c = np.concatenate(verts).mean(axis=0)
            i = np.argmin((glon - c[0]) ** 2 + (glat - c[1]) ** 2)
            w.flat[i] = 1.0
        out[f["properties"]["iso3"]] = w
    return out


# --------------------------------------------------------------------------- CPC climate
def read_cpc(path: Path, var: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(dates, lat, lon, values[time, lat, lon]) from a NetCDF-3 subset; missing values as NaN."""
    from scipy.io import netcdf_file

    with netcdf_file(path, mmap=False) as nc:
        t = nc.variables["time"]
        origin = pd.Timestamp(t.units.decode().split("since")[1].strip())
        dates = (origin + pd.to_timedelta(t[:].astype(float), unit="h")).normalize().to_numpy()
        lat, lon = nc.variables["lat"][:].astype(float), nc.variables["lon"][:].astype(float)
        x = nc.variables[var]
        v = x[:].astype(float)
        bad = np.isclose(v, float(x.missing_value)) | (np.abs(v) > 1e20)
    v[bad] = np.nan
    return dates, lat, lon, v


def country_daily(dates, values: np.ndarray, weights: dict[str, np.ndarray], var: str) -> pd.DataFrame:
    """Area-weighted mean over each country's cells with data, per day."""
    rows = []
    for iso3, w in weights.items():
        ok = ~np.isnan(values) & (w > 0)[None]
        num = np.nansum(np.where(ok, values * w[None], 0.0), axis=(1, 2))
        den = np.sum(np.where(ok, w[None], 0.0), axis=(1, 2))
        with np.errstate(invalid="ignore"):
            rows.append(pd.DataFrame({"iso3": iso3, "date": dates, var: num / den,
                                      f"{var}_cells": ok.sum(axis=(1, 2))}))
    return pd.concat(rows, ignore_index=True)


def cpc_year(var: str, year: int, weights_for, cache: Path, refresh: bool) -> pd.DataFrame:
    path = cache / f"{var}.{year}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date"])
    q = {"var": var, **CPC_BOX, "horizStride": 1, "temporal": "all", "accept": "netcdf"}
    url = CPC_URL.format(path=CPC_VARS[var], year=year) + "?" + urllib.parse.urlencode(q)
    with tempfile.TemporaryDirectory() as tmp:
        nc = Path(tmp) / "subset.nc"
        nc.write_bytes(_get(url))
        dates, lat, lon, v = read_cpc(nc, var)
    df = country_daily(dates, v, weights_for(lat, lon), var)
    df.to_csv(path, index=False)
    return df


def cpc_climate(out_dir: Path, geojson: Path, first_year: int = CPC_FIRST_YEAR) -> Path:
    """Country x day table of precipitation (mm) and tmax / tmin (degC), first_year to today."""
    cache = out_dir / CPC_CACHE
    cache.mkdir(exist_ok=True)
    this_year = dt.date.today().year
    memo = {}

    def weights_for(lat, lon):
        key = (lat.tobytes(), lon.tobytes())
        if key not in memo:
            memo[key] = country_weights(geojson, lat, lon)
        return memo[key]

    frames = {}
    for var in CPC_VARS:
        parts = []
        for year in range(first_year, this_year + 1):
            log.info("  CPC %s %d", var, year)
            parts.append(cpc_year(var, year, weights_for, cache, refresh=year == this_year))
        frames[var] = pd.concat(parts, ignore_index=True)
    out = frames["precip"]
    for var in ("tmax", "tmin"):
        out = out.merge(frames[var], on=["iso3", "date"], how="outer")
    path = out_dir / CPC_FILE
    out = out.sort_values(["iso3", "date"])
    out[["precip", "tmax", "tmin"]] = out[["precip", "tmax", "tmin"]].round(3)
    out.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SOURCE_DIR, help="folder for the downloads (default: SOURCE_DIR)")
    parser.add_argument("--skip-climate", action="store_true", help="only World Bank and boundaries (fast)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("World Bank WDI -> %s", world_bank(args.out))
    geo = boundaries(args.out)
    log.info("Natural Earth boundaries -> %s", geo)
    if not args.skip_climate:
        log.info("NOAA CPC daily climate (about 72 requests, ~35 min the first time)")
        log.info("  -> %s", cpc_climate(args.out, geo))


if __name__ == "__main__":
    main()
