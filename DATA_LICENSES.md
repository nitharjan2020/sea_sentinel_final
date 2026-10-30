# Data licences and attribution

The MIT licence in `LICENSE` covers the code, tests, documentation and figures written by the
SEA-Sentinel team. It does **not** cover third-party data. The source files in `data/raw/`, and the
tables derived from them in `data/processed/` and `outputs/tables/`, stay under their providers'
terms, listed below. They are included only so the analysis can be reproduced exactly. Anyone
reusing them must follow the original terms.

Every raw file is listed with its SHA-256 checksum and retrieval date in `data/raw/MANIFEST.csv`.

| File(s) in `data/raw/` | Provider | Terms |
|---|---|---|
| `empres_i/empres_i_sea_hpai_asf_rabies_2026-10-21.csv` | FAO, EMPRES-i Plus | FAO database terms of use, CC BY 4.0 (see below) |
| `faostat/faostat_qcl_stocks_sea_2026-10-20.csv` | FAO, FAOSTAT (QCL) | FAO database terms of use, CC BY 4.0 (see below) |
| `glw/glw_census_metadata.xlsx` | FAO, Gridded Livestock of the World | Terms in each layer's metadata |
| `wahis/wahis_event_reports_*.csv` | WOAH, WAHIS | WOAH terms and conditions (see below) |
| `worldbank/worldbank_wdi_sea_2026-10-24.csv` | World Bank, World Development Indicators | CC BY 4.0 (see below) |
| `cpc/cpc_climate_sea_country_daily_2026-10-24.csv` | NOAA CPC via NOAA PSL (country averages computed by `download.py`) | US government data, public domain; acknowledgement requested (see below) |
| `naturalearth/naturalearth_admin0_sea.geojson` | Natural Earth | Public domain |

## FAO (EMPRES-i Plus, FAOSTAT, GLW)

FAO statistical databases and datasets are licensed under Creative Commons Attribution 4.0
International (CC BY 4.0). The terms require attribution and forbid using the data to promote a
commercial enterprise or to misrepresent its content. Some EMPRES-i records come from third parties
(WOAH, national authorities, media), as each row's `Diagnosis.source` shows.
Terms: <https://www.fao.org/contact-us/terms/db-terms-of-use/en/>

Required citations:

- FAO. 2026. EMPRES-i Plus – Global Animal Disease Information System. Accessed on 21 October 2026.
  https://empres-i.apps.fao.org/ Licence: CC-BY-4.0.
- FAO. 2026. FAOSTAT: Crops and livestock products (QCL). Accessed on 20 October 2026.
  https://www.fao.org/faostat/en/#data/QCL Licence: CC-BY-4.0.

## WOAH (WAHIS)

The data belong to the reporting national authorities and are published by WOAH through WAHIS
(<https://wahis.woah.org/>). WOAH's terms allow reuse when the information is unaltered and the
source is acknowledged. Commercial use, and copying or transmitting the information to third
parties, need WOAH's prior written permission.
Terms: <https://www.woah.org/en/terms-and-conditions/>

The two WAHIS event-list extracts are included unaltered, solely for the WOAH International
Datathon 2026 submission to WOAH. If this repository is later made public, these two files should
be removed, or WOAH's permission obtained first; `README.md` explains how to re-download them.

Citation: World Organisation for Animal Health (WOAH). 2026. World Animal Health Information
System (WAHIS), event list. Accessed on 20 October 2026. https://wahis.woah.org/

## World Bank (World Development Indicators)

Licensed under Creative Commons Attribution 4.0 (CC BY 4.0).
Terms: <https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets>

Citation: World Bank. 2026. World Development Indicators. Accessed on 24 October 2026 through the
World Bank API. https://databank.worldbank.org/ Licence: CC BY 4.0.

## NOAA CPC (climate)

CPC Global Unified gauge-based analysis of daily precipitation, and CPC global daily maximum and
minimum temperature, 0.5 degree. US government data with no restrictions on use; NOAA PSL asks for
this acknowledgement:

"CPC Global Unified Precipitation and Temperature data provided by the NOAA PSL, Boulder, Colorado,
USA, from their website at https://psl.noaa.gov." Accessed on 24 October 2026.

The file in `data/raw/cpc/` is not the original grid: `download.py` averages each day's grid over
each country's land cells (Natural Earth outlines) at download time.

## Natural Earth

Public domain. "Made with Natural Earth. Free vector and raster map data @ naturalearthdata.com."

