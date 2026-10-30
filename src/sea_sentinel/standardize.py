"""Harmonise raw records to ISO3 country codes, project disease codes, species groups and dates.

Every input row is kept in the output with an ``include`` flag and an ``exclude_reason`` so that the
coverage audit can account for what was dropped and why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COUNTRY_ALIASES


def to_iso3(name: object) -> str | None:
    if not isinstance(name, str):
        return None
    return COUNTRY_ALIASES.get(name.strip().lower())


def week_start(dates: pd.Series) -> pd.Series:
    """Monday of the ISO week containing each date."""
    return (dates - pd.to_timedelta(dates.dt.day_of_week, unit="D")).dt.normalize()


# --------------------------------------------------------------------------- EMPRES-i
EMPRES_DISEASE = {
    "Influenza - Avian": "HPAI",
    "African swine fever": "ASF",
}  # anything else (e.g. rabies, dropped from scope) is excluded as "disease outside scope"


def _species_group(animal_type: pd.Series) -> pd.Series:
    a = animal_type.fillna("").str.lower()
    return pd.Series(
        np.select(
            [a.str.contains("domestic"), a.str.contains("wild"), a.str.contains("captive"), a == ""],
            ["domestic", "wild", "captive", "unknown"],
            default="other",
        ),
        index=animal_type.index,
    )


def _exclusion_reason(df: pd.DataFrame) -> pd.Series:
    """First applicable reason a record is out of scope, or <NA> if it is in scope."""
    a = df["animal_type"].fillna("").str.lower()
    sero = df["serotype"].fillna("").str.upper()
    hpai, asf = (df["disease"] == d for d in ("HPAI", "ASF"))
    rules = [
        (df["iso3"].isna(), "country outside scope"),
        (df["disease"].isna(), "disease outside scope"),
        (df["diagnosis_status"].str.lower().ne("confirmed").fillna(True), "not confirmed"),
        (hpai & sero.str.contains("LPAI"), "low pathogenicity serotype"),
        (hpai & df["species_group"].isin(["wild", "captive"]), "non-poultry bird"),
        (asf & ~(a.str.contains("domestic") & a.str.contains("swine|pig")), "non-domestic swine"),
        (df["event_date"].isna(), "no usable date"),
    ]
    reason = pd.Series(pd.NA, index=df.index, dtype="object")
    for mask, label in reversed(rules):  # earlier rules take precedence
        reason = reason.mask(mask, label)
    return reason


def standardize_empres(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "source": "EMPRES-i",
            "source_id": "SRC003",
            "record_id": raw["Event.ID"],
            "global_id": raw["Global.ID"],
            "iso3": raw["Country"].map(to_iso3),
            "country_raw": raw["Country"],
            "disease": raw["Disease"].map(EMPRES_DISEASE),
            "disease_raw": raw["Disease"],
            "serotype": raw["Serotype"],
            "admin1": raw["Admin.level.1"],
            "locality": raw["Locality"],
            "latitude": pd.to_numeric(raw["Latitude"], errors="coerce"),
            "longitude": pd.to_numeric(raw["Longitude"], errors="coerce"),
            "reporting_source": raw["Diagnosis.source"],
            "diagnosis_status": raw["Diagnosis.status"],
            "animal_type": raw["Animal.type"],
            "species": raw["Species"],
            "observation_date": pd.to_datetime(
                raw["Observation.date..dd.mm.yyyy."], format="%d/%m/%Y", errors="coerce"
            ),
            "report_date": pd.to_datetime(
                raw["Report.date..dd.mm.yyyy."], format="%d/%m/%Y", errors="coerce"
            ),
        }
    )
    df["species_group"] = _species_group(df["animal_type"])

    # Analysis date: observation (onset/detection) date; fall back to report date, flagged.
    df["date_basis"] = np.where(
        df["observation_date"].notna(),
        "observation",
        np.where(df["report_date"].notna(), "report", "none"),
    )
    df["event_date"] = df["observation_date"].fillna(df["report_date"])

    delay = (df["report_date"] - df["observation_date"]).dt.days
    df["report_delay_days"] = delay.astype("Int64")
    # Negative delays are data errors; very long ones are typically retrospective batch uploads.
    df["delay_quality"] = np.select(
        [delay.isna(), delay < 0, delay > 365],
        ["missing", "negative", "over_1y"],
        default="ok",
    )

    df["exclude_reason"] = _exclusion_reason(df)
    df["include"] = df["exclude_reason"].isna()
    df["event_week"] = week_start(df["event_date"])
    df["report_week"] = week_start(df["report_date"])
    return df


# --------------------------------------------------------------------------- WAHIS
def _wahis_disease(text: pd.Series) -> pd.Series:
    t = text.fillna("").str.lower()
    return pd.Series(
        np.select(
            [
                t.str.contains("high pathogenicity avian influenza") & t.str.contains("poultry"),
                t.str.contains("african swine fever"),
            ],
            ["HPAI", "ASF"],
            default=None,
        ),
        index=text.index,
    )


def standardize_wahis(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per WAHIS event, keeping the latest report seen for that event.

    These are *events* (which may contain many outbreaks) and must not be added to outbreak counts.
    The submission date belongs to the listed report, so a notification delay is only defined when
    that report is the immediate notification (``IN``).
    """
    df = pd.DataFrame(
        {
            "source": "WAHIS",
            "source_id": "SRC001",
            "event_id": raw["eventId"],
            "report_id": raw["reportId"],
            "iso3": raw["countryTerritory"].map(to_iso3),
            "country_raw": raw["countryTerritory"],
            "disease": _wahis_disease(raw["disease"]),
            "disease_raw": raw["disease"],
            "subtype": raw["subType"],
            "reason": raw["reason"],
            "report_status": raw["reportStatus"],
            "report_number": raw["reportNumber"],
            "event_start_date": pd.to_datetime(raw["eventStartDate"], format="%m/%d/%Y", errors="coerce"),
            "submission_date": pd.to_datetime(raw["submissionDate"], format="%m/%d/%Y", errors="coerce"),
            "source_file": raw["source_file"],
        }
    )
    df = (
        df[df["disease"].notna()]  # events of diseases outside scope (e.g. rabies) are dropped
        .sort_values("submission_date")
        .drop_duplicates("event_id", keep="last")
        .sort_values(["iso3", "disease", "event_start_date"])
        .reset_index(drop=True)
    )
    is_in = df["report_number"].eq("IN")
    df["notification_delay_days"] = (
        (df["submission_date"] - df["event_start_date"]).dt.days.where(is_in).astype("Int64")
    )
    return df


# --------------------------------------------------------------------------- FAOSTAT
FAOSTAT_UNIT_MULTIPLIER = {"1000 An": 1000, "An": 1}


GLW_ITEMS = {"Chicken": "chickens", "Swine": "swine_pigs"}  # GLW sheet -> FAOSTAT item name


def standardize_glw_census(raw: pd.DataFrame) -> pd.DataFrame:
    """National census totals behind GLW: iso3, item, census_year, census_head, census_source.

    Repairs the workbook's quirks: a mis-encoded apostrophe in Laos' name, numbers stored as text
    with thousands separators and tabs, and -9999 / blanks for countries without a census.
    """
    df = raw.rename(columns=lambda c: str(c).strip())
    out = pd.DataFrame({
        "iso3": df["Country"].str.replace("�", "'", regex=False).map(to_iso3),
        "item": df["sheet"].map(GLW_ITEMS),
        "census_year": pd.to_numeric(df["Year"], errors="coerce"),
        "census_head": pd.to_numeric(df["Census"].str.replace(r"[,\s]", "", regex=True), errors="coerce"),
        "census_source": df["Source"].str.strip(),
    })
    out = out[out["iso3"].notna() & out["item"].notna() & out["census_year"].notna() & (out["census_head"] > 0)]
    return out.astype({"census_year": int}).sort_values(["item", "iso3"]).reset_index(drop=True)


def standardize_faostat(raw: pd.DataFrame) -> pd.DataFrame:
    """Tidy annual stocks in head of animals: iso3, year, item, head_count."""
    df = raw[raw["Element"].eq("Stocks")].copy()
    df["iso3"] = df["Area"].map(to_iso3)
    df = df[df["iso3"].notna()]
    head_count = pd.to_numeric(df["Value"], errors="coerce") * df["Unit"].map(FAOSTAT_UNIT_MULTIPLIER)
    out = pd.DataFrame(
        {
            "iso3": df["iso3"],
            "year": pd.to_numeric(df["Year"]).astype(int),
            "item": df["Item"].str.lower().str.replace(" / ", "_").str.replace(" ", "_"),
            "head_count": head_count,
            "flag": df["Flag"],
            "flag_description": df["Flag Description"],
        }
    )
    return out.sort_values(["iso3", "item", "year"]).reset_index(drop=True)
