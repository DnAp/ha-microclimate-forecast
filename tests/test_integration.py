"""The integration inside a real Home Assistant: flow, setup, entities, forecasts."""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from homeassistant.config_entries import SOURCE_USER
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.microclimate_forecast.const import (
    CONF_HUMIDITY,
    CONF_LEARNING_DAYS,
    CONF_TEMPERATURE,
    DOMAIN,
)
from custom_components.microclimate_forecast.open_meteo import FORECAST_URL, PREVIOUS_RUNS_URL

FIXTURES = Path(__file__).parent / "fixtures"
TZ = ZoneInfo("Asia/Nicosia")
NOW = datetime(2026, 10, 18, 20, 30, tzinfo=UTC)  # 23:30 in Cyprus


def night(local_hour: int) -> bool:
    return local_hour < 8 or local_hour >= 21


def station_history() -> dict[datetime, float]:
    """The station = yesterday's forecast, 3.5° colder at night and 0.7° warmer by day."""
    prev = json.loads((FIXTURES / "previous.json").read_text())
    fc = {
        datetime.fromtimestamp(t, tz=UTC): v
        for t, v in zip(prev["hourly"]["time"], prev["hourly"]["temperature_2m_previous_day1"], strict=False)
    }
    out = {}
    for t, v in fc.items():
        nxt = fc.get(t + timedelta(hours=1))
        if nxt is None or t >= NOW - timedelta(hours=1):
            continue
        mid = (t + timedelta(minutes=30)).astimezone(TZ)
        out[t] = (v + nxt) / 2 + (-3.5 if night(mid.hour) else 0.7)
    return out


@pytest.fixture
async def setup_hass(hass: HomeAssistant, aioclient_mock, freezer):
    freezer.move_to(NOW)
    await hass.config.async_set_time_zone("Asia/Nicosia")
    hass.config.latitude, hass.config.longitude = 34.92, 33.63
    hass.states.async_set("sensor.station_temp", "12.3", {"unit_of_measurement": "°C", "device_class": "temperature"})
    hass.states.async_set("sensor.station_hum", "92", {"unit_of_measurement": "%", "device_class": "humidity"})
    aioclient_mock.get(FORECAST_URL, text=(FIXTURES / "forecast.json").read_text())
    aioclient_mock.get(PREVIOUS_RUNS_URL, text=(FIXTURES / "previous.json").read_text())
    with patch(
        "custom_components.microclimate_forecast.coordinator.async_hourly_means",
        return_value=station_history(),
    ):
        yield hass


async def create_entry(hass: HomeAssistant):
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    assert result["type"] is FlowResultType.FORM
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "name": "Yard",
            CONF_TEMPERATURE: "sensor.station_temp",
            CONF_HUMIDITY: "sensor.station_hum",
            CONF_LEARNING_DAYS: 30,
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    return result["result"]


async def test_setup_learns_and_corrects(setup_hass: HomeAssistant):
    hass = setup_hass
    await create_entry(hass)

    state = hass.states.get("weather.yard")
    assert state is not None
    # Now = the station, not the model.
    assert state.attributes["temperature"] == 12.3
    assert state.attributes["humidity"] == 92
    assert state.attributes["bias_learned"] is True
    assert state.attributes["bias_by_hour"]["04"] == pytest.approx(-3.5, abs=0.1)
    assert state.attributes["bias_by_hour"]["13"] == pytest.approx(0.7, abs=0.1)
    assert state.attributes["mae_per_hour"] < state.attributes["mae_raw"]
    assert state.attributes["method"] == "analogs"
    assert state.attributes["analog_segments"] > 60
    assert state.attributes["mae_analogs"] < state.attributes["mae_raw"]

    raw = json.loads((FIXTURES / "forecast.json").read_text())["hourly"]
    raw_by_time = {
        datetime.fromtimestamp(t, tz=UTC): v for t, v in zip(raw["time"], raw["temperature_2m"], strict=False)
    }

    resp = await hass.services.async_call(
        "weather", "get_forecasts", {"entity_id": "weather.yard", "type": "hourly"}, blocking=True, return_response=True
    )
    hourly = resp["weather.yard"]["forecast"]
    assert datetime.fromisoformat(hourly[0]["datetime"]) >= NOW.replace(minute=0)
    four_am = next(h for h in hourly if datetime.fromisoformat(h["datetime"]).astimezone(TZ).hour == 4)
    t = datetime.fromisoformat(four_am["datetime"])
    # raw + the station's night offset (analogs all say −3.5) + the 23:30 deviation, faded over 4.5 h.
    faded = state.attributes["nowcast_offset"] * math.exp(-(t - NOW).total_seconds() / 3600 / 4)
    assert four_am["temperature"] == pytest.approx(raw_by_time[t] - 3.5 + faded, abs=0.3)
    assert "nowcast_offset" in state.attributes

    resp = await hass.services.async_call(
        "weather", "get_forecasts", {"entity_id": "weather.yard", "type": "daily"}, blocking=True, return_response=True
    )
    daily = resp["weather.yard"]["forecast"]
    assert len(daily) >= 6
    assert all(d["templow"] < d["temperature"] for d in daily)

    low = hass.states.get("sensor.yard_night_low")
    assert low is not None
    assert float(low.state) < low.attributes["uncorrected_forecast"] - 3
    high = hass.states.get("sensor.yard_day_high")
    assert float(high.state) > high.attributes["uncorrected_forecast"]


async def test_station_change_updates_now(setup_hass: HomeAssistant):
    hass = setup_hass
    await create_entry(hass)
    hass.states.async_set("sensor.station_temp", "11.0", {"unit_of_measurement": "°C"})
    await hass.async_block_till_done()
    assert hass.states.get("weather.yard").attributes["temperature"] == 11.0


async def test_fahrenheit_station(setup_hass: HomeAssistant):
    hass = setup_hass
    hass.states.async_set("sensor.station_temp", "50", {"unit_of_measurement": "°F"})
    await create_entry(hass)
    # HA shows in the system unit (°C here): 50 °F = 10 °C.
    assert hass.states.get("weather.yard").attributes["temperature"] == 10.0


async def test_duplicate_sensor_aborts(setup_hass: HomeAssistant):
    hass = setup_hass
    await create_entry(hass)
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"name": "Again", CONF_TEMPERATURE: "sensor.station_temp", CONF_LEARNING_DAYS: 30}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_options_clear_humidity_and_reload(setup_hass: HomeAssistant):
    hass = setup_hass
    entry = await create_entry(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_TEMPERATURE: "sensor.station_temp", CONF_LEARNING_DAYS: 14}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    state = hass.states.get("weather.yard")
    # Humidity now comes from the model, not the cleared station sensor.
    assert state.attributes["humidity"] != 92


async def test_forecast_outage_is_unavailable_not_crash(hass: HomeAssistant, aioclient_mock, freezer):
    freezer.move_to(NOW)
    hass.states.async_set("sensor.station_temp", "12.3", {"unit_of_measurement": "°C"})
    aioclient_mock.get(FORECAST_URL, status=503, text="{}")
    aioclient_mock.get(PREVIOUS_RUNS_URL, status=503, text="{}")
    with patch("custom_components.microclimate_forecast.coordinator.async_hourly_means", return_value={}):
        result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"name": "Yard", CONF_TEMPERATURE: "sensor.station_temp", CONF_LEARNING_DAYS: 30}
        )
        await hass.async_block_till_done()
    entry = hass.config_entries.async_entries(DOMAIN)[0]
    assert entry.state.name == "SETUP_RETRY"
