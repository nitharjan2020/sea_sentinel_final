"""Project-wide constants: paths, country codes, disease definitions and source registry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Folder holding the original downloads (defaults to the folder containing this repo).
SOURCE_DIR = Path(os.environ.get("SEA_SENTINEL_SOURCE_DIR", PROJECT_ROOT.parent))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# --------------------------------------------------------------------------- scope
# ISO3 -> display name, in the order used throughout the project.
COUNTRIES: dict[str, str] = {
    "BRN": "Brunei",
    "KHM": "Cambodia",
    "IDN": "Indonesia",
    "LAO": "Laos",
    "MYS": "Malaysia",
    "MMR": "Myanmar",
    "PHL": "Philippines",
    "SGP": "Singapore",
    "THA": "Thailand",
    "TLS": "Timor-Leste",
    "VNM": "Vietnam",
}

# Every spelling seen in the raw sources -> ISO3.
COUNTRY_ALIASES: dict[str, str] = {
    "brunei": "BRN",
    "brunei darussalam": "BRN",
    "cambodia": "KHM",
    "indonesia": "IDN",
    "laos": "LAO",
    "lao pdr": "LAO",
    "lao people's democratic republic": "LAO",
    "lao people’s democratic republic": "LAO",
    "malaysia": "MYS",
    "myanmar": "MMR",
    "philippines": "PHL",
    "singapore": "SGP",
    "thailand": "THA",
    "timor-leste": "TLS",
    "east timor": "TLS",
    "vietnam": "VNM",
    "viet nam": "VNM",
}

DISEASES: dict[str, str] = {
    "HPAI": "Highly pathogenic avian influenza (poultry)",
    "ASF": "African swine fever (domestic swine)",
}
TARGET_SPECIES: dict[str, str] = {"HPAI": "Poultry", "ASF": "Swine"}
# Rabies was in the original scope but was dropped: dog rabies records are sparse and there are no dog
# population data. Its rows stay in the raw EMPRES-i/WAHIS snapshots and are excluded as out of scope.


# --------------------------------------------------------------------------- sources
@dataclass(frozen=True)
class RawSource:
    source_id: str          # matches the Source Register in the dataset inventory
    original_name: str      # file name as downloaded into SOURCE_DIR
    raw_name: str           # canonical name under data/raw/
    provider: str
    url: str
    retrieved: str          # YYYY-MM-DD
    unit: str               # what one row represents
    licence_note: str


RAW_SOURCES: tuple[RawSource, ...] = (
    RawSource(
        source_id="SRC003",
        # NB: downloaded with the misleading name "FAOSTAT.csv"; the header shows it is EMPRES-i Plus.
        original_name="FAOSTAT.csv",
        raw_name="empres_i/empres_i_sea_hpai_asf_rabies_2026-10-21.csv",
        provider="FAO EMPRES-i Plus",
        url="https://empres-i.apps.fao.org/",
        retrieved="2026-10-21",
        unit="outbreak (location x observation date)",
        licence_note="FAO database terms of use: CC BY 4.0 with attribution",
    ),
    RawSource(
        source_id="SRC001",
        original_name="WAHIS.csv",
        raw_name="wahis/wahis_event_reports_recent_2026-10-20.csv",
        provider="WOAH WAHIS",
        url="https://wahis.woah.org/",
        retrieved="2026-10-20",
        unit="event report (one row per latest report of an event)",
        licence_note="WOAH terms and conditions; not covered by the MIT licence",
    ),
    RawSource(
        source_id="SRC001",
        original_name="Event-list-10_20_2026 (1).csv",
        raw_name="wahis/wahis_event_reports_historical_2026-10-20.csv",
        provider="WOAH WAHIS",
        url="https://wahis.woah.org/",
        retrieved="2026-10-20",
        unit="event report (one row per latest report of an event)",
        licence_note="WOAH terms and conditions; not covered by the MIT licence",
    ),
    RawSource(
        source_id="SRC004",
        original_name="FAOSTAT_data_en_10-20-2026.csv",
        raw_name="faostat/faostat_qcl_stocks_sea_2026-10-20.csv",
        provider="FAOSTAT (QCL)",
        url="https://www.fao.org/faostat/en/#data",
        retrieved="2026-10-20",
        unit="country x item x year livestock stock",
        licence_note="FAO database terms of use: CC BY 4.0 with attribution",
    ),
    RawSource(
        source_id="SRC005",
        original_name="GRIDED FAO DATA.xlsx",
        raw_name="glw/glw_census_metadata.xlsx",
        provider="FAO Gridded Livestock of the World (census metadata)",
        url="https://www.fao.org/livestock-systems/global-distributions/en/",
        retrieved="2026-10-22",
        unit="country census summary (chicken sheet, pig sheet)",
        licence_note="FAO GLW; terms per dataset metadata",
    ),
    # Open covariate sources, fetched by `python -m sea_sentinel.download` into SOURCE_DIR.
    RawSource(
        source_id="SRC007",
        original_name="worldbank_wdi_sea.csv",
        raw_name="worldbank/worldbank_wdi_sea_2026-10-24.csv",
        provider="World Bank, World Development Indicators",
        url="https://api.worldbank.org/v2/ (WDI; DataBank https://databank.worldbank.org/)",
        retrieved="2026-10-24",
        unit="country x indicator x year (population by sex, land area, density, GDP, trade)",
        licence_note="World Bank open data: CC BY 4.0 with attribution",
    ),
    RawSource(
        source_id="SRC011",
        original_name="cpc_climate_sea_country_daily.csv",
        raw_name="cpc/cpc_climate_sea_country_daily_2026-10-24.csv",
        provider="NOAA CPC Global Unified gauge-based daily precipitation and temperature (via NOAA PSL)",
        url="https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html ; .../data.cpc.globaltemp.html",
        retrieved="2026-10-24",
        unit="country x day: 0.5-degree grid averaged over each country's land cells (download.py)",
        licence_note="NOAA data: public domain; acknowledge NOAA PSL and CPC",
    ),
    RawSource(
        source_id="SRC012",
        original_name="naturalearth_admin0_sea.geojson",
        raw_name="naturalearth/naturalearth_admin0_sea.geojson",
        provider="Natural Earth, admin-0 countries 1:50m (11 countries kept)",
        url="https://www.naturalearthdata.com/",
        retrieved="2026-10-24",
        unit="country boundary (used only to average the climate grids)",
        licence_note="Public domain",
    ),
)

# --------------------------------------------------------------------------- analysis
WEEK_FREQ = "W-SUN"  # ISO-style weeks: Monday start, labelled by their Monday below
LONG_GAP_WEEKS = 26  # silent stretch after which a series is treated as a surveillance gap

# Cross-source duplicate matching, chosen from the lag/distance review (python -m sea_sentinel.dedup_review):
# the cross- vs same-source pair ratio is elevated only up to ~1 km and ~3 days.
DEDUP_MAX_KM = 1.0
DEDUP_WINDOW_DAYS = 3

# FAOSTAT stock items forming each disease's at-risk population (offset).
POPULATION_ITEMS: dict[str, tuple[str, ...]] = {
    "HPAI": ("chickens", "ducks", "geese", "turkeys"),
    "ASF": ("swine_pigs",),
}
# FAOSTAT publishes stocks 1-2 years late, so the stock of year Y is anchored at 1 July of Y + lag: a week
# only uses values that had (about) been published, in detection and in all its backtests.
POPULATION_LAG_YEARS = 2

# Forecast eligibility thresholds (applied to the deduplicated, dated outbreak series).
ELIGIBILITY = {
    "min_dated_outbreaks": 100,
    "min_active_weeks": 104,
    "min_nonzero_weeks": 30,
}
