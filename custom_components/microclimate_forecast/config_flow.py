"""Config and options flow: pick the station's sensors."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.const import CONF_NAME
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    CONF_DEW_POINT,
    CONF_HUMIDITY,
    CONF_LEARNING_DAYS,
    CONF_PRESSURE,
    CONF_TEMPERATURE,
    CONF_UV,
    CONF_WIND_BEARING,
    CONF_WIND_GUST,
    CONF_WIND_SPEED,
    DEFAULT_LEARNING_DAYS,
    DEFAULT_NAME,
    DOMAIN,
    MAX_LEARNING_DAYS,
    MIN_LEARNING_DAYS,
    OPTIONAL_SENSORS,
)

DEVICE_CLASSES: dict[str, list[str] | None] = {
    CONF_TEMPERATURE: [SensorDeviceClass.TEMPERATURE],
    CONF_HUMIDITY: [SensorDeviceClass.HUMIDITY],
    CONF_PRESSURE: [SensorDeviceClass.PRESSURE, SensorDeviceClass.ATMOSPHERIC_PRESSURE],
    CONF_WIND_SPEED: [SensorDeviceClass.WIND_SPEED, SensorDeviceClass.SPEED],
    CONF_WIND_GUST: [SensorDeviceClass.WIND_SPEED, SensorDeviceClass.SPEED],
    CONF_WIND_BEARING: None,
    CONF_DEW_POINT: [SensorDeviceClass.TEMPERATURE],
    CONF_UV: None,
}


def _entity(key: str) -> selector.EntitySelector:
    classes = DEVICE_CLASSES[key]
    cfg = (
        selector.EntitySelectorConfig(domain="sensor", device_class=classes)
        if classes
        else selector.EntitySelectorConfig(domain="sensor")
    )
    return selector.EntitySelector(cfg)


def _sensor_schema(defaults: dict[str, Any]) -> dict[Any, Any]:
    schema: dict[Any, Any] = {
        vol.Required(CONF_TEMPERATURE, default=defaults.get(CONF_TEMPERATURE, vol.UNDEFINED)): _entity(
            CONF_TEMPERATURE
        ),
    }
    for key in OPTIONAL_SENSORS:
        if defaults.get(key):
            schema[vol.Optional(key, description={"suggested_value": defaults[key]})] = _entity(key)
        else:
            schema[vol.Optional(key)] = _entity(key)
    schema[vol.Required(CONF_LEARNING_DAYS, default=defaults.get(CONF_LEARNING_DAYS, DEFAULT_LEARNING_DAYS))] = (
        selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=MIN_LEARNING_DAYS,
                max=MAX_LEARNING_DAYS,
                step=1,
                mode=selector.NumberSelectorMode.BOX,
                unit_of_measurement="d",
            )
        )
    )
    return schema


class MicroclimateConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input is not None:
            await self.async_set_unique_id(user_input[CONF_TEMPERATURE])
            self._abort_if_unique_id_configured()
            name = user_input.pop(CONF_NAME, DEFAULT_NAME)
            user_input[CONF_LEARNING_DAYS] = int(user_input[CONF_LEARNING_DAYS])
            return self.async_create_entry(title=name, data=user_input)

        schema = {vol.Required(CONF_NAME, default=DEFAULT_NAME): selector.TextSelector(), **_sensor_schema({})}
        return self.async_show_form(step_id="user", data_schema=vol.Schema(schema))

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return MicroclimateOptionsFlow()


class MicroclimateOptionsFlow(OptionsFlow):
    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input is not None:
            user_input[CONF_LEARNING_DAYS] = int(user_input[CONF_LEARNING_DAYS])
            # Every optional sensor absent from the form was cleared, not kept.
            for key in OPTIONAL_SENSORS:
                user_input.setdefault(key, None)
            new_sensor = user_input[CONF_TEMPERATURE]
            if new_sensor != self.config_entry.unique_id:
                others = [
                    e
                    for e in self.hass.config_entries.async_entries(DOMAIN)
                    if e.entry_id != self.config_entry.entry_id and e.unique_id == new_sensor
                ]
                if others:
                    return self.async_abort(reason="already_configured")
                self.hass.config_entries.async_update_entry(self.config_entry, unique_id=new_sensor)
            return self.async_create_entry(data=user_input)
        current = {**self.config_entry.data, **self.config_entry.options}
        return self.async_show_form(step_id="init", data_schema=vol.Schema(_sensor_schema(current)))
