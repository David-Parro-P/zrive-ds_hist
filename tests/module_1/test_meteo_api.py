from unittest.mock import Mock
from src.module_1.module_1_meteo_api import (
    f_construye_params,
    f_verifica_esquema_basic,
    f_check_codes,
    get_data_meteo_api,
    API_URL,
    COORDINATES,
    VARIABLES,
    MODELS,
)


"""
Ejemplos de algunos tests que podemos hacer

pytest /home/dparro/zrive-ds/tests/module_1/test_meteo_api.py
==================================================================================== test session starts =====================================================================================
platform linux -- Python 3.11.0, pytest-7.4.0, pluggy-1.2.0
rootdir: /home/dparro/zrive-ds
plugins: mock-3.14.0
collected 4 items                                                                                                                                                                            

tests/module_1/test_meteo_api.py ....                                                                                                                                                  [100%]

===================================================================================== 4 passed in 0.45s ======================================================================================
"""


def test_f_construye_params():
    cities = {
        "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
        "London": {"latitude": 51.507351, "longitude": -0.127758},
    }
    start_date = "2023-01-01"
    end_date = "2023-02-01"
    data = "temperature_2m_mean"
    models = ["CMCC_CM2_VHR4"]
    expected = [
        {
            "latitude": 40.416775,
            "longitude": -3.703790,
            "start_date": start_date,
            "end_date": end_date,
            "models": models,
            "daily": data,
        },
        {
            "latitude": 51.507351,
            "longitude": -0.127758,
            "start_date": start_date,
            "end_date": end_date,
            "models": models,
            "daily": data,
        },
    ]
    result = f_construye_params(cities, start_date, end_date, data, models)
    assert result == expected


def test_f_verifica_esquema_basic():
    correct_packet = {
        "latitude": 40.416775,
        "longitude": -3.703790,
        "utc_offset_seconds": 3600,
        "timezone": "CET",
        "timezone_abbreviation": "CET",
        "elevation": 667,
        "daily_units": "metric",
    }
    assert f_verifica_esquema_basic(correct_packet)

    incorrect_packet = {"latitude": 40.416775, "longitude": -3.703790}
    assert not f_verifica_esquema_basic(incorrect_packet)


def test_f_check_codes(mocker):
    mock_response = Mock()
    mock_response.status_code = 200
    assert f_check_codes(mock_response)

    mock_response.status_code = 400
    assert not f_check_codes(mock_response)


def test_get_data_meteo_api(mocker):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "latitude": 40.416775,
        "longitude": -3.703790,
        "utc_offset_seconds": 3600,
        "timezone": "CET",
        "timezone_abbreviation": "CET",
        "elevation": 667,
        "daily_units": "metric",
    }
    mocker.patch("requests.get", return_value=mock_response)
    result = get_data_meteo_api(
        API_URL, COORDINATES, "2023-01-01", "2023-02-01", VARIABLES, MODELS
    )
    assert isinstance(result, list)
    assert len(result) > 0
