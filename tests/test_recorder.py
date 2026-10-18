"""Reading the station's hourly history from a real recorder."""

from datetime import UTC, datetime, timedelta

import pytest
from homeassistant.components.recorder.models import StatisticMeanType
from homeassistant.components.recorder.statistics import async_import_statistics
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.components.recorder.common import async_wait_recording_done

from custom_components.microclimate_forecast.coordinator import async_hourly_means


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(recorder_db_url, enable_custom_integrations):
    """The recorder database must exist before `hass` is set up."""
    yield


HOUR0 = datetime(2026, 10, 10, 0, tzinfo=UTC)


async def test_long_term_statistics(recorder_mock, hass: HomeAssistant):
    hass.states.async_set("sensor.station_temp", "12", {"unit_of_measurement": "°C", "state_class": "measurement"})
    async_import_statistics(
        hass,
        {
            "mean_type": StatisticMeanType.ARITHMETIC,
            "has_sum": False,
            "name": None,
            "source": "recorder",
            "statistic_id": "sensor.station_temp",
            "unit_class": "temperature",
            "unit_of_measurement": "°C",
        },
        [{"start": HOUR0 + timedelta(hours=h), "mean": 10.0 + h, "min": 9.0, "max": 30.0} for h in range(5)],
    )
    await async_wait_recording_done(hass)
    out = await async_hourly_means(hass, "sensor.station_temp", HOUR0, HOUR0 + timedelta(hours=5))
    assert out == {HOUR0 + timedelta(hours=h): 10.0 + h for h in range(5)}


async def test_falls_back_to_state_history(recorder_mock, hass: HomeAssistant, freezer):
    # No state_class → no long-term statistics; raw states are averaged per hour.
    for minute, value in ((5, 10.0), (35, 12.0), (65, 20.0)):
        freezer.move_to(HOUR0 + timedelta(minutes=minute))
        hass.states.async_set("sensor.plain_temp", str(value), {"unit_of_measurement": "°C"})
        await hass.async_block_till_done()
    await async_wait_recording_done(hass)
    freezer.move_to(HOUR0 + timedelta(hours=3))
    out = await async_hourly_means(hass, "sensor.plain_temp", HOUR0, HOUR0 + timedelta(hours=2))
    assert out == {HOUR0: 11.0, HOUR0 + timedelta(hours=1): 20.0}
