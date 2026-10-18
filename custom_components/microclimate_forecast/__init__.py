"""Microclimate Forecast: an Open-Meteo forecast corrected by your own station."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .coordinator import MicroclimateCoordinator

PLATFORMS = [Platform.WEATHER, Platform.SENSOR]

type MicroclimateConfigEntry = ConfigEntry[MicroclimateCoordinator]


async def async_setup_entry(hass: HomeAssistant, entry: MicroclimateConfigEntry) -> bool:
    coordinator = MicroclimateCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = coordinator
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_reload))
    return True


async def _async_reload(hass: HomeAssistant, entry: MicroclimateConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: MicroclimateConfigEntry) -> bool:
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_entry(hass: HomeAssistant, entry: MicroclimateConfigEntry) -> None:
    """Drop the learned bias with the entry."""
    from homeassistant.helpers.storage import Store

    from .const import DOMAIN, STORAGE_VERSION

    await Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}").async_remove()
