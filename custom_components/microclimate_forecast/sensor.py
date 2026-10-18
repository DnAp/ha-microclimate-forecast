"""Convenience sensors for automations: the coming night's low, the coming day's high."""

from __future__ import annotations

from datetime import UTC
from typing import Any

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .const import DAY_END_HOUR, DAY_START_HOUR, NIGHT_END_HOUR, NIGHT_START_HOUR
from .coordinator import MicroclimateCoordinator
from .model import window_extreme
from .weather import device_info

WINDOWS = (
    # translation key, window start hour, end hour (local), min or max
    ("night_low", NIGHT_START_HOUR, NIGHT_END_HOUR, "min"),
    ("day_high", DAY_START_HOUR, DAY_END_HOUR, "max"),
)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    coordinator: MicroclimateCoordinator = entry.runtime_data
    async_add_entities(WindowSensor(coordinator, entry, *w) for w in WINDOWS)


class WindowSensor(CoordinatorEntity[MicroclimateCoordinator], SensorEntity):
    """Lowest/highest corrected temperature of the current-or-next local window."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS

    def __init__(
        self, coordinator: MicroclimateCoordinator, entry: ConfigEntry, key: str, start: int, end: int, pick: str
    ) -> None:
        super().__init__(coordinator)
        self._start, self._end, self._pick = start, end, pick
        self._attr_translation_key = key
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_device_info = device_info(entry)

    def _extremes(self) -> tuple[tuple[float, Any] | None, float | None]:
        data = self.coordinator.data
        if not data:
            return None, None
        tz = dt_util.get_time_zone(self.hass.config.time_zone) or UTC
        now = dt_util.utcnow()
        corrected = window_extreme(data.hourly, now, tz, self._start, self._end, self._pick)
        raw_points = [
            type(p)(time=p.time, temperature=p.raw_temperature, raw_temperature=p.raw_temperature) for p in data.hourly
        ]
        raw = window_extreme(raw_points, now, tz, self._start, self._end, self._pick)
        return corrected, (raw[0] if raw else None)

    @property
    def native_value(self) -> float | None:
        corrected, _ = self._extremes()
        return corrected[0] if corrected else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        corrected, raw = self._extremes()
        if not corrected:
            return {}
        return {"at": corrected[1].isoformat(), "uncorrected_forecast": raw}
