import json

import numpy as np
import pandas as pd
import pytest
from scipy.io import netcdf_file

from sea_sentinel import covariates as cov
from sea_sentinel import download as dl
from sea_sentinel.forecast import Series


def daily(days=6 * 365, start="2016-01-04", iso3="VNM", seed=0):
    """Daily climate with a seasonal cycle and noise."""
    rng = np.random.default_rng(seed)
    d = pd.date_range(start, periods=days, freq="D")
    season = np.sin(2 * np.pi * d.day_of_year.to_numpy() / 365.25)
    return pd.DataFrame({"iso3": iso3, "date": d, "tmax": 30 + 3 * season + rng.normal(0, 1, days),
                         "tmin": 22 + 3 * season + rng.normal(0, 1, days),
                         "precip": rng.gamma(1.0, 5 + 4 * season.clip(0), days)})


def test_climate_weekly_removes_the_season_and_lags_only_past_weeks(monkeypatch):
    monkeypatch.setattr(cov, "CLIMATOLOGY_YEARS", (2016, 2019))
    w = cov.climate_weekly(daily())
    ref = w[w["week"].dt.year <= 2019]
    assert abs(ref["temp_anom"].mean()) < 0.1 and abs(ref["precip_anom"].mean()) < 0.1
    i = w.index[40]
    assert w.loc[i, "temp_anom_4w"] == pytest.approx(w.loc[i - 3:i, "temp_anom"].mean(), abs=1e-3)
    assert w["temp_anom_4w"].iloc[:3].isna().all()  # needs 4 past weeks


def test_partial_weeks_are_dropped_and_weekly_precip_is_a_total():
    d = daily(days=24)  # starts on a Monday: 3 full weeks + 3 days
    w = cov.climate_weekly(d)
    assert len(w) == 3
    first = d.iloc[:7]
    assert w["precip"].iloc[0] == pytest.approx(first["precip"].sum(), rel=1e-3)
    assert w["tmean"].iloc[0] == pytest.approx(((first["tmax"] + first["tmin"]) / 2).mean(), abs=1e-3)


def test_climate_matrix_aligns_to_panel_weeks():
    w = cov.climate_weekly(daily())
    weeks = pd.Series(pd.date_range("2015-12-28", periods=10, freq="W-MON"))  # first week before the data
    C = cov.climate_matrix(w, "VNM", weeks)
    assert C.shape == (10, 2) and np.isnan(C[0]).all()
    assert np.isnan(cov.climate_matrix(w, "THA", weeks)).all()


def test_socioeconomic_density_by_sex():
    wb = pd.DataFrame({"iso3": "VNM", "year": 2020, "variable": ["population_female", "population_male",
                                                                 "land_area_km2"], "value": [500.0, 400.0, 10.0]})
    t = cov.socioeconomic(wb)
    assert t.loc[0, "population_density_female_km2"] == 50 and t.loc[0, "population_density_male_km2"] == 40


def test_country_weights_inside_and_nearest_cell_fallback(tmp_path):
    square = [[[100.0, 10.0], [102.0, 10.0], [102.0, 12.0], [100.0, 12.0], [100.0, 10.0]]]
    tiny = [[[105.1, 1.1], [105.2, 1.1], [105.2, 1.2], [105.1, 1.2], [105.1, 1.1]]]  # no cell centre inside
    geo = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"iso3": "THA"}, "geometry": {"type": "Polygon", "coordinates": square}},
        {"type": "Feature", "properties": {"iso3": "SGP"}, "geometry": {"type": "Polygon", "coordinates": tiny}}]}
    path = tmp_path / "b.geojson"
    path.write_text(json.dumps(geo))
    lat, lon = np.arange(14.75, -0.1, -0.5), np.arange(99.25, 106.0, 0.5)
    w = dl.country_weights(path, lat, lon)
    assert (w["THA"] > 0).sum() == 16  # 4 x 4 cell centres in a 2 x 2 degree square
    assert (w["SGP"] > 0).sum() == 1
    vals = np.ones((2, len(lat), len(lon)))
    vals[:, w["THA"] > 0] = 5.0
    vals[1, np.nonzero(w["THA"])[0][0], np.nonzero(w["THA"])[1][0]] = np.nan  # one missing cell on day 2
    out = dl.country_daily(np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]"), vals, w, "precip")
    tha = out[out["iso3"] == "THA"]
    assert np.allclose(tha["precip"], 5.0) and list(tha["precip_cells"]) == [16, 15]


def test_read_cpc_converts_time_and_missing_values(tmp_path):
    path = tmp_path / "x.nc"
    with netcdf_file(path, "w") as nc:
        nc.createDimension("time", 2)
        nc.createDimension("lat", 1)
        nc.createDimension("lon", 2)
        t = nc.createVariable("time", "d", ("time",))
        t.units = b"hours since 1900-01-01 00:00:00"
        t[:] = [24 * (pd.Timestamp("2024-01-01") - pd.Timestamp("1900-01-01")).days + h for h in (0, 24)]
        nc.createVariable("lat", "f", ("lat",))[:] = [10.25]
        nc.createVariable("lon", "f", ("lon",))[:] = [100.25, 100.75]
        v = nc.createVariable("precip", "f", ("time", "lat", "lon"))
        v.missing_value = np.float32(-9.96921e36)
        v[:] = np.array([[[1.0, -9.96921e36]], [[2.0, 3.0]]], dtype=np.float32)
    dates, lat, lon, vals = dl.read_cpc(path, "precip")
    assert list(pd.to_datetime(dates).strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02"]
    assert np.isnan(vals[0, 0, 1]) and vals[1, 0, 1] == 3.0


def series_with(climate):
    rng = np.random.default_rng(2)
    weeks = pd.Series(pd.date_range("2016-01-04", periods=260, freq="W-MON"))
    return Series(weeks, pd.Series(rng.poisson(4, 260).astype(float)), climate=climate)


def test_climate_variant_appears_only_with_climate_and_matches_nb_ar_when_uninformative():
    assert "nb_ar_climate" not in series_with(None).params(200, 2)
    p = series_with(np.zeros((260, 2))).params(200, 2)
    assert p["nb_ar_climate"][1] == pytest.approx(p["nb_ar"][1], rel=1e-3)


def test_climate_variant_does_not_look_ahead():
    C = np.random.default_rng(4).normal(size=(260, 2))
    a = series_with(C).params(200, 3)["nb_ar_climate"]
    C2 = C.copy()
    C2[201:] = 50.0  # change the climate after the origin
    assert series_with(C2).params(200, 3)["nb_ar_climate"] == a
