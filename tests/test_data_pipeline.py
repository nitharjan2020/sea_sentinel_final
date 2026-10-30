import pandas as pd
import pytest

from sea_sentinel.coverage import eligibility
from sea_sentinel.dedup import deduplicate
from sea_sentinel.denominators import add_population, annual_population, census_check
from sea_sentinel.load import repair_empres_row
from sea_sentinel.panel import build_weekly_panel
from sea_sentinel.standardize import standardize_empres, standardize_glw_census, to_iso3, week_start

EMPRES_COLS = [
    "Event.ID", "Global.ID", "Disease", "Serotype", "Region", "Subregion", "Country", "Admin.level.1",
    "Locality", "Latitude", "Longitude", "Diagnosis.source", "Diagnosis.status", "Animal.type", "Species",
    "Observation.date..dd.mm.yyyy.", "Report.date..dd.mm.yyyy.", "Humans.affected", "Human.deaths",
]


def empres_row(**kw):
    base = dict(zip(EMPRES_COLS, [pd.NA] * len(EMPRES_COLS)))
    base.update({
        "Event.ID": "1", "Global.ID": "G1", "Disease": "Influenza - Avian", "Serotype": "H5N1 HPAI",
        "Country": "Viet Nam", "Admin.level.1": "Ha Tinh", "Locality": "A", "Latitude": "18.0",
        "Longitude": "105.0", "Diagnosis.source": "National authorities", "Diagnosis.status": "Confirmed",
        "Animal.type": "Domestic - chicken", "Species": "Domestic - Chicken",
        "Observation.date..dd.mm.yyyy.": "01/03/2024", "Report.date..dd.mm.yyyy.": "05/03/2024",
    })
    base.update(kw)
    return base


def std(*rows):
    return standardize_empres(pd.DataFrame(list(rows), columns=EMPRES_COLS))


# --------------------------------------------------------------------------- load
def test_repair_unquoted_comma_in_species():
    row = ["x"] * 11 + ["WOAH (former OIE)", "Confirmed", "Wild - Suid (A", " B )", "Wild - Suid (A", " B )",
                        "09/08/2026", "03/09/2026", "", ""]
    fixed = repair_empres_row(row, 19)
    assert len(fixed) == 19
    assert fixed[13] == fixed[14] == "Wild - Suid (A, B )"
    assert fixed[15] == "09/08/2026"


def test_repair_unquoted_comma_in_source():
    row = ["x"] * 11 + ["National Institute", " NICD", "Confirmed", "Domestic - Swine", "Domestic - Swine",
                        "04/04/2022", "12/04/2022", "", ""]
    fixed = repair_empres_row(row, 19)
    assert fixed[11:15] == ["National Institute, NICD", "Confirmed", "Domestic - Swine", "Domestic - Swine"]


# --------------------------------------------------------------------------- standardize
@pytest.mark.parametrize("name,iso", [("Viet Nam", "VNM"), ("Lao People's Democratic Republic", "LAO"),
                                      ("Brunei Darussalam", "BRN"), ("China", None)])
def test_to_iso3(name, iso):
    assert to_iso3(name) == iso


def test_scope_rules():
    df = std(
        empres_row(),
        empres_row(Serotype="H9N2 LPAI"),
        empres_row(**{"Animal.type": "Wild - mallard"}),
        empres_row(**{"Diagnosis.status": "Suspected"}),
        empres_row(Disease="African swine fever", **{"Animal.type": "Wild - Wild boar"}),
        empres_row(Disease="Rabies", **{"Animal.type": "Domestic - Dogs"}),
    )
    assert df["include"].tolist() == [True, False, False, False, False, False]
    assert df.loc[1, "exclude_reason"] == "low pathogenicity serotype"
    assert df.loc[5, "exclude_reason"] == "disease outside scope"  # rabies was dropped from scope


def test_report_date_fallback_is_flagged():
    df = std(empres_row(**{"Observation.date..dd.mm.yyyy.": pd.NA}))
    assert df.loc[0, "date_basis"] == "report"
    assert df.loc[0, "event_date"] == pd.Timestamp("2024-03-05")
    assert pd.isna(df.loc[0, "report_delay_days"])


def test_week_start_is_monday():
    s = week_start(pd.Series(pd.to_datetime(["2024-03-03", "2024-03-04", "2024-03-10"])))
    assert s.tolist() == [pd.Timestamp("2024-02-26"), pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-04")]


# --------------------------------------------------------------------------- dedup
def test_exact_repeats_and_species_rows_collapse():
    df = std(
        empres_row(**{"Event.ID": "1"}),
        empres_row(**{"Event.ID": "2"}),                                     # re-upload
        empres_row(**{"Event.ID": "3", "Animal.type": "Domestic - duck"}),  # species row, same outbreak
        empres_row(**{"Event.ID": "4", "Locality": "B", "Latitude": "18.5"}),  # elsewhere
    )
    out = deduplicate(df[df["include"]])
    assert len(out) == 2
    merged = out[out["n_merged"] == 3].iloc[0]
    assert set(merged["animal_types"].split("|")) == {"Domestic - chicken", "Domestic - duck"}


def test_cross_source_merge_keeps_national_record():
    df = std(
        empres_row(**{"Event.ID": "1", "Diagnosis.source": "WOAH (former OIE)", "Locality": "a village"}),
        empres_row(**{"Event.ID": "2", "Observation.date..dd.mm.yyyy.": "03/03/2024", "Latitude": "18.001"}),
        empres_row(**{"Event.ID": "3", "Observation.date..dd.mm.yyyy.": "20/03/2024"}),  # weeks later
    )
    out = deduplicate(df[df["include"]])
    assert len(out) == 2
    merged = out[out["n_merged"] == 2].iloc[0]
    assert merged["record_id"] == "2" and merged["reporting_source"] == "National authorities"
    assert merged["reporting_sources"] == "National authorities|WOAH (former OIE)"


def test_cross_source_uses_true_distance_not_grid_cells():
    # ~300 m apart but on opposite sides of a 0.01-degree grid line
    df = std(
        empres_row(**{"Event.ID": "1", "Latitude": "18.0049"}),
        empres_row(**{"Event.ID": "2", "Latitude": "18.0076", "Diagnosis.source": "WOAH (former OIE)"}),
        empres_row(**{"Event.ID": "3", "Latitude": "18.03", "Diagnosis.source": "Media"}),  # ~2.8 km away
    )
    out = deduplicate(df[df["include"]], max_km=1.0)
    assert sorted(out["n_merged"]) == [1, 2]


def test_same_source_neighbours_are_never_merged():
    # Two national records 100 m apart are distinct outbreaks, even when a WOAH record sits between them.
    df = std(
        empres_row(**{"Event.ID": "1", "Locality": "A"}),
        empres_row(**{"Event.ID": "2", "Locality": "B", "Latitude": "18.0009"}),
        empres_row(**{"Event.ID": "3", "Locality": "C", "Latitude": "18.0005", "Diagnosis.source": "WOAH (former OIE)"}),
    )
    out = deduplicate(df[df["include"]])
    assert len(out) == 2
    assert all(len(set(s.split("|"))) == len(s.split("|")) for s in out["reporting_sources"])


# --------------------------------------------------------------------------- panel
def test_panel_missing_is_not_zero():
    df = std(
        empres_row(**{"Event.ID": "1", "Observation.date..dd.mm.yyyy.": "06/03/2024"}),
        empres_row(**{"Event.ID": "2", "Observation.date..dd.mm.yyyy.": "27/03/2024", "Locality": "B"}),
    )
    out = deduplicate(df[df["include"]])
    panel = build_weekly_panel(out, end_week=pd.Timestamp("2024-04-15"),
                               start_weeks={"HPAI": pd.Timestamp("2024-02-19")})
    s = panel[(panel["iso3"] == "VNM") & (panel["disease"] == "HPAI")].set_index("week")
    assert s.loc["2024-02-19", "coverage_status"] == "before_first_report"
    assert pd.isna(s.loc["2024-02-19", "outbreaks"])
    assert s.loc["2024-03-04", "outbreaks"] == 1
    assert s.loc["2024-03-11", "outbreaks"] == 0          # reported zero inside the window
    assert s.loc["2024-04-15", "coverage_status"] == "after_last_report"
    assert pd.isna(s.loc["2024-04-15", "outbreaks"])
    # a country with no records is entirely missing
    assert panel[(panel["iso3"] == "BRN")]["outbreaks"].isna().all()


# --------------------------------------------------------------------------- population offsets
def test_population_interpolation_and_flags():
    stocks = pd.DataFrame({
        "iso3": ["VNM"] * 4 + ["SGP"],
        "year": [2020, 2021, 2020, 2021, 2020],
        "item": ["chickens", "chickens", "ducks", "ducks", "chickens"],
        "head_count": [100.0, 400.0, 0.0, 0.0, 50.0],
        "flag": ["A", "E", "A", "A", "A"],
    })
    annual = annual_population(stocks)
    vnm = annual[(annual["iso3"] == "VNM") & (annual["disease"] == "HPAI")].set_index("year")
    assert vnm.loc[2021, "population_head"] == 400 and vnm.loc[2021, "pct_official"] == 0
    assert annual[annual["disease"] == "ASF"].empty  # no pig stocks -> no population, not zero

    weeks = pd.to_datetime(["2020-01-06", "2021-01-04", "2022-01-03"])
    panel = pd.DataFrame({"iso3": "VNM", "disease": ["HPAI"] * 3 + ["ASF"] * 3, "week": list(weeks) * 2})
    out = add_population(panel, annual)
    h = out[out["disease"] == "HPAI"].set_index("week")
    assert h.loc["2020-01-06", "population_head"] == 100 and h.loc["2020-01-06", "population_extrapolated"]
    assert 190 < h.loc["2021-01-04", "population_head"] < 210   # geometric midpoint of 100 and 400
    assert not h.loc["2021-01-04", "population_extrapolated"]
    assert h.loc["2022-01-03", "population_head"] == 400 and h.loc["2022-01-03", "population_extrapolated"]
    assert out.loc[out["disease"] == "ASF", "log_population"].isna().all()

    lagged = add_population(panel, annual, lag_years=1).set_index(["disease", "week"])
    assert lagged.loc[("HPAI", pd.Timestamp("2021-01-04")), "population_head"] == 100



def test_glw_census_quirks_and_check_against_faostat():
    raw = pd.DataFrame({
        "Country": ["Lao People�s Democratic Republic", "Philippines", "Singapore", "Viet Nam"],
        "Year": ["2009.0", "2006.0", None, "2009.0"],
        "Census": ["16047773", "134,115,530	", "-9999", "21786059"],
        "Source": ["DLF", "BAI", "-9999", "GSO"],
        "sheet": ["Chicken", "Chicken", "Chicken", "Swine"],
    })
    c = standardize_glw_census(raw)
    assert list(zip(c["iso3"], c["item"])) == [("LAO", "chickens"), ("PHL", "chickens"), ("VNM", "swine_pigs")]
    assert c.loc[c["iso3"] == "PHL", "census_head"].iloc[0] == 134_115_530  # text with comma and tab
    stocks = pd.DataFrame({"iso3": ["LAO", "PHL"], "year": [2009, 2006], "item": ["chickens", "chickens"],
                           "head_count": [8_000_000.0, 134_115_530.0], "flag": ["I", "A"]})
    chk = census_check(c, stocks).set_index("iso3")
    assert round(chk.loc["LAO", "ratio"], 2) == 2.01 and chk.loc["PHL", "ratio"] == 1
    assert pd.isna(chk.loc["VNM", "ratio"])  # no FAOSTAT pigs in this toy table: ratio missing, row kept


def test_eligibility_tiers():
    assert eligibility(500, 500, 300, 100).startswith("Eligible")
    assert eligibility(50, 500, 300, 20).startswith("Detection")
    assert eligibility(0, 4, 1, 1).startswith("Descriptive")
    assert eligibility(0, 0, 0, 0).startswith("Withhold")
