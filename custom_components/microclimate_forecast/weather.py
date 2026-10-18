"""Weather entity: the station's own readings now, the corrected forecast ahead."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from homeassistant.components.weather import (
    Forecast,
    SingleCoordinatorWeatherEntity,
    WeatherEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    UnitOfPrecipitationDepth,
    UnitOfPressure,
    UnitOfSpeed,
    UnitOfTemperature,
)
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import PressureConverter, SpeedConverter, TemperatureConverter

from .const import (
    CONF_DEW_POINT,
    CONF_HUMIDITY,
    CONF_PRESSURE,
    CONF_TEMPERATURE,
    CONF_UV,
    CONF_WIND_BEARING,
    CONF_WIND_GUST,
    CONF_WIND_SPEED,
    DOMAIN,
)
from .coordinator import MicroclimateCoordinator
from .model import HourPoint, condition_for

ATTRIBUTION = "Forecast: Open-Meteo.com, corrected by this station's history"


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    async_add_entities([MicroclimateWeather(entry.runtime_data, entry)])


def device_info(entry: ConfigEntry) -> DeviceInfo:
    return DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=entry.title,
        manufacturer="Open-Meteo + local station",
        entry_type=DeviceEntryType.SERVICE,
    )


class MicroclimateWeather(SingleCoordinatorWeatherEntity[MicroclimateCoordinator]):
    _attr_has_entity_name = True
    _attr_name = None
    _attr_attribution = ATTRIBUTION
    _attr_native_temperature_unit = UnitOfTemperature.CELSIUS
    _attr_native_pressure_unit = UnitOfPressure.HPA
    _attr_native_wind_speed_unit = UnitOfSpeed.KILOMETERS_PER_HOUR
    _attr_native_precipitation_unit = UnitOfPrecipitationDepth.MILLIMETERS
    _attr_supported_features = WeatherEntityFeature.FORECAST_HOURLY | WeatherEntityFeature.FORECAST_DAILY

    def __init__(self, coordinator: MicroclimateCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = device_info(entry)

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        tracked = [e for e in self._sensor_ids() if e]
        if tracked:
            self.async_on_remove(async_track_state_change_event(self.hass, tracked, self._on_sensor_change))

    @callback
    def _on_sensor_change(self, _event: Event[EventStateChangedData]) -> None:
        self.async_write_ha_state()

    # ── Station readings ──

    def _sensor_ids(self) -> list[str | None]:
        s = self.coordinator.settings
        keys = (
            CONF_TEMPERATURE,
            CONF_HUMIDITY,
            CONF_PRESSURE,
            CONF_WIND_SPEED,
            CONF_WIND_GUST,
            CONF_WIND_BEARING,
            CONF_DEW_POINT,
            CONF_UV,
        )
        return [s.get(k) for k in keys]

    def _reading(self, key: str) -> tuple[float, str | None] | None:
        entity_id = self.coordinator.settings.get(key)
        if not entity_id or (state := self.hass.states.get(entity_id)) is None:
            return None
        try:
            return float(state.state), state.attributes.get("unit_of_measurement")
        except (TypeError, ValueError):
            return None

    def _temp(self, key: str) -> float | None:
        r = self._reading(key)
        if r is None:
            return None
        value, unit = r
        if unit and unit != UnitOfTemperature.CELSIUS and unit in TemperatureConverter.VALID_UNITS:
            value = TemperatureConverter.convert(value, unit, UnitOfTemperature.CELSIUS)
        return round(value, 1)

    def _speed(self, key: str) -> float | None:
        r = self._reading(key)
        if r is None:
            return None
        value, unit = r
        if unit and unit in SpeedConverter.VALID_UNITS:
            value = SpeedConverter.convert(value, unit, UnitOfSpeed.KILOMETERS_PER_HOUR)
        return round(value, 1)

    def _now_point(self) -> HourPoint | None:
        data = self.coordinator.data
        if not data or not data.hourly:
            return None
        now = dt_util.utcnow()
        return min(data.hourly, key=lambda p: abs((p.time - now).total_seconds()))

    @property
    def condition(self) -> str | None:
        p = self._now_point()
        return condition_for(p.weather_code, p.is_day) if p else None

    @property
    def native_temperature(self) -> float | None:
        station = self._temp(CONF_TEMPERATURE)
        if station is not None:
            return station
        p = self._now_point()
        return p.temperature if p else None

    @property
    def native_dew_point(self) -> float | None:
        station = self._temp(CONF_DEW_POINT)
        if station is not None:
            return station
        p = self._now_point()
        return p.dew_point if p else None

    @property
    def humidity(self) -> float | None:
        r = self._reading(CONF_HUMIDITY)
        if r is not None:
            return r[0]
        p = self._now_point()
        return p.humidity if p else None

    @property
    def native_pressure(self) -> float | None:
        r = self._reading(CONF_PRESSURE)
        if r is not None:
            value, unit = r
            if unit and unit in PressureConverter.VALID_UNITS:
                value = PressureConverter.convert(value, unit, UnitOfPressure.HPA)
            return round(value, 1)
        p = self._now_point()
        return p.pressure if p else None

    @property
    def native_wind_speed(self) -> float | None:
        station = self._speed(CONF_WIND_SPEED)
        if station is not None:
            return station
        p = self._now_point()
        return p.wind_speed if p else None

    @property
    def native_wind_gust_speed(self) -> float | None:
        station = self._speed(CONF_WIND_GUST)
        if station is not None:
            return station
        p = self._now_point()
        return p.wind_gust if p else None

    @property
    def wind_bearing(self) -> float | None:
        r = self._reading(CONF_WIND_BEARING)
        if r is not None:
            return r[0]
        p = self._now_point()
        return p.wind_bearing if p else None

    @property
    def uv_index(self) -> float | None:
        r = self._reading(CONF_UV)
        if r is not None:
            return r[0]
        p = self._now_point()
        return p.uv_index if p else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        bias = self.coordinator.bias
        attrs: dict[str, Any] = {"bias_learned": bias is not None}
        if bias is not None:
            attrs.update(
                {
                    "bias_by_hour": {f"{h:02d}": v for h, v in enumerate(bias.hours)},
                    "learned_at": bias.learned_at.isoformat(),
                    "learning_days": bias.days,
                    "mae_raw": bias.mae_raw,
                    "mae_corrected": bias.mae_corrected,
                }
            )
        data = self.coordinator.data
        if data and data.extra.get("nowcast_offset") is not None:
            attrs["nowcast_offset"] = data.extra["nowcast_offset"]
        if self.coordinator.learn_error:
            attrs["learn_error"] = self.coordinator.learn_error
        return attrs

    # ── Forecasts ──

    @callback
    def _async_forecast_hourly(self) -> list[Forecast] | None:
        data = self.coordinator.data
        if not data:
            return None
        cutoff = dt_util.utcnow().replace(minute=0, second=0, microsecond=0)
        out: list[Forecast] = []
        for p in data.hourly:
            if p.time < cutoff:
                continue
            out.append(
                Forecast(
                    datetime=p.time.isoformat(),
                    condition=condition_for(p.weather_code, p.is_day),
                    native_temperature=p.temperature,
                    native_dew_point=p.dew_point,
                    humidity=p.humidity,
                    cloud_coverage=p.cloud_cover,
                    native_precipitation=p.precipitation,
                    precipitation_probability=p.precipitation_probability,
                    native_pressure=p.pressure,
                    native_wind_speed=p.wind_speed,
                    native_wind_gust_speed=p.wind_gust,
                    wind_bearing=p.wind_bearing,
                    uv_index=p.uv_index,
                    is_daytime=p.is_day,
                )
            )
        return out

    @callback
    def _async_forecast_daily(self) -> list[Forecast] | None:
        data = self.coordinator.data
        if not data:
            return None
        tz = dt_util.get_time_zone(self.hass.config.time_zone) or UTC
        today = dt_util.now(tz).date()
        out: list[Forecast] = []
        for d in data.daily:
            if d.day < today:
                continue
            out.append(
                Forecast(
                    datetime=datetime(d.day.year, d.day.month, d.day.day, tzinfo=tz).isoformat(),
                    condition=condition_for(d.weather_code, True),
                    native_temperature=d.temperature,
                    native_templow=d.templow,
                    native_precipitation=d.precipitation,
                    precipitation_probability=d.precipitation_probability,
                    native_wind_speed=d.wind_speed,
                    native_wind_gust_speed=d.wind_gust,
                    wind_bearing=d.wind_bearing,
                    uv_index=d.uv_index,
                )
            )
        return out
