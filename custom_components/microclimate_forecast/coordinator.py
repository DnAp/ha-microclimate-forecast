"""Coordinator: hourly forecast fetch, daily re-learning of the station's bias."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import TimestampDataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import TemperatureConverter

from .const import (
    CONF_LEARNING_DAYS,
    CONF_TEMPERATURE,
    DEFAULT_LEARNING_DAYS,
    DOMAIN,
    FORECAST_DAYS,
    RELEARN_INTERVAL,
    STORAGE_VERSION,
    UPDATE_INTERVAL,
)
from .model import (
    AnalogModel,
    BiasModel,
    ForecastData,
    apply_corrections,
    apply_nowcast,
    build_past_segments,
    daily_from_hourly,
    learn_analogs,
    learn_bias,
    pair_samples,
)
from .open_meteo import OpenMeteoError, fetch_forecast, fetch_past_forecasts

_LOGGER = logging.getLogger(__name__)


async def async_hourly_means(
    hass: HomeAssistant, entity_id: str, start: datetime, end: datetime
) -> dict[datetime, float]:
    """Hourly mean °C of a sensor, keyed by the UTC start of the hour.

    Long-term statistics first: they outlive the recorder's purge window (10 days by
    default). A sensor without a state_class has none — then raw state history.
    """
    from homeassistant.components.recorder import get_instance, history, statistics

    rec = get_instance(hass)
    stats = await rec.async_add_executor_job(
        statistics.statistics_during_period,
        hass,
        start,
        end,
        {entity_id},
        "hour",
        {"temperature": UnitOfTemperature.CELSIUS},
        {"mean"},
    )
    out: dict[datetime, float] = {}
    for row in stats.get(entity_id, []):
        mean = row.get("mean")
        if mean is None:
            continue
        begin = row["start"]
        when = datetime.fromtimestamp(begin, tz=UTC) if isinstance(begin, (int, float)) else begin
        out[when] = float(mean)
    if out:
        return out

    states = await rec.async_add_executor_job(history.state_changes_during_period, hass, start, end, entity_id, True)
    buckets: dict[datetime, list[float]] = defaultdict(list)
    for st in states.get(entity_id, []):
        try:
            value = float(st.state)
        except (TypeError, ValueError):
            continue
        when = st.last_changed.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        buckets[when].append(value)
    unit = None
    if (cur := hass.states.get(entity_id)) is not None:
        unit = cur.attributes.get("unit_of_measurement")
    to_c = (lambda v: (v - 32) * 5 / 9) if unit == UnitOfTemperature.FAHRENHEIT else (lambda v: v)
    return {k: to_c(sum(v) / len(v)) for k, v in buckets.items()}


class MicroclimateCoordinator(TimestampDataUpdateCoordinator[ForecastData]):
    """Fetches the forecast every hour and corrects it with the learned bias."""

    config_entry: ConfigEntry

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=f"{DOMAIN} {entry.title}",
            update_interval=UPDATE_INTERVAL,
        )
        self._store: Store[dict[str, Any]] = Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}")
        self.bias: BiasModel | None = None
        self.analogs: AnalogModel | None = None
        self.learn_error: str | None = None
        self._loaded = False

    @property
    def settings(self) -> dict[str, Any]:
        return {**self.config_entry.data, **self.config_entry.options}

    @property
    def learning_days(self) -> int:
        return int(self.settings.get(CONF_LEARNING_DAYS, DEFAULT_LEARNING_DAYS))

    def _coords(self) -> tuple[float, float]:
        return float(self.hass.config.latitude), float(self.hass.config.longitude)

    async def _async_load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        stored = await self._store.async_load()
        if not stored:
            return
        try:
            if stored.get("bias"):
                self.bias = BiasModel.from_dict(stored["bias"])
            if stored.get("analogs"):
                self.analogs = AnalogModel.from_dict(stored["analogs"])
        except (KeyError, TypeError, ValueError):
            # A store from an older version without analogs re-learns at the first update.
            _LOGGER.warning("Stored model is unreadable; it will be re-learned")
            self.bias = self.analogs = None

    async def async_learn(self) -> None:
        """Re-learn the bias from the station's history. Keeps the old one on failure."""
        tz = dt_util.get_time_zone(self.hass.config.time_zone) or UTC
        now = dt_util.utcnow()
        days = self.learning_days
        start = (now - timedelta(days=days)).replace(minute=0, second=0, microsecond=0)
        # The last full hour — the current one is still being measured.
        end = now.replace(minute=0, second=0, microsecond=0)
        lat, lon = self._coords()
        try:
            observed = await async_hourly_means(self.hass, self.settings[CONF_TEMPERATURE], start, end)
            past = await fetch_past_forecasts(async_get_clientsession(self.hass), lat, lon, days + 1)
        except OpenMeteoError as err:
            self.learn_error = f"past forecasts: {err}"
            _LOGGER.warning("Could not fetch past forecasts: %s", err)
            return
        samples = pair_samples(observed, {t: h.temperature for t, h in past.items()}, tz)
        model = learn_bias(samples, now)
        if model is None:
            self.learn_error = "no station history overlapping the past forecasts yet"
            _LOGGER.info("No samples to learn from yet (%s observed hours)", len(observed))
            return
        self.bias = model
        self.analogs = learn_analogs(build_past_segments(observed, past, tz), now)
        self.learn_error = None
        await self._store.async_save(
            {"bias": model.as_dict(), "analogs": self.analogs.as_dict() if self.analogs else None}
        )
        _LOGGER.debug(
            "Learned over %s days: per-hour MAE %.2f → %.2f, analogs → %s",
            model.days,
            model.mae_raw or 0,
            model.mae_corrected or 0,
            self.analogs.mae_corrected if self.analogs else None,
        )

    async def _async_update_data(self) -> ForecastData:
        await self._async_load()
        now = dt_util.utcnow()
        stale = self.bias is None or now - self.bias.learned_at >= RELEARN_INTERVAL
        if stale or self.analogs is None:
            try:
                await self.async_learn()
            except Exception as err:  # noqa: BLE001 — a learning failure must not cost the forecast
                self.learn_error = str(err)
                _LOGGER.exception("Learning the bias failed")

        tz = dt_util.get_time_zone(self.hass.config.time_zone) or UTC
        lat, lon = self._coords()
        try:
            hourly, daily_meta = await fetch_forecast(
                async_get_clientsession(self.hass), lat, lon, self.hass.config.time_zone, FORECAST_DAYS
            )
        except OpenMeteoError as err:
            raise UpdateFailed(f"Open-Meteo forecast: {err}") from err
        analog_points = apply_corrections(hourly, self.analogs, self.bias, tz)
        offset = apply_nowcast(hourly, self._station_now(), now)
        return ForecastData(
            hourly=hourly,
            daily=daily_from_hourly(hourly, daily_meta, tz),
            bias=self.bias,
            fetched_at=now,
            extra={"nowcast_offset": offset, "analog_points": analog_points},
        )

    def _station_now(self) -> float | None:
        """The station's current reading in °C, or None if it has none."""
        state = self.hass.states.get(self.settings[CONF_TEMPERATURE])
        if state is None:
            return None
        try:
            value = float(state.state)
        except (TypeError, ValueError):
            return None
        unit = state.attributes.get("unit_of_measurement")
        if unit and unit != UnitOfTemperature.CELSIUS and unit in TemperatureConverter.VALID_UNITS:
            value = TemperatureConverter.convert(value, unit, UnitOfTemperature.CELSIUS)
        return value
