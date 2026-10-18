"""Thin Open-Meteo client: the current forecast and the archive of past forecasts."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import aiohttp

from .model import HourPoint

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
TIMEOUT = aiohttp.ClientTimeout(total=30)

HOURLY = (
    "temperature_2m,relative_humidity_2m,dew_point_2m,weather_code,is_day,cloud_cover,"
    "wind_speed_10m,wind_gusts_10m,wind_direction_10m,precipitation,"
    "precipitation_probability,pressure_msl,uv_index"
)
DAILY = (
    "weather_code,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,"
    "wind_gusts_10m_max,wind_direction_10m_dominant,uv_index_max"
)


class OpenMeteoError(Exception):
    """The API did not return usable data."""


async def _get(session: aiohttp.ClientSession, url: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        async with session.get(url, params=params, timeout=TIMEOUT) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200 or not isinstance(data, dict) or data.get("error"):
                reason = data.get("reason") if isinstance(data, dict) else None
                raise OpenMeteoError(f"HTTP {resp.status}: {reason or 'bad response'}")
            return data
    except (aiohttp.ClientError, TimeoutError) as err:
        raise OpenMeteoError(str(err)) from err


def _utc(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)


def _col(block: dict[str, Any], name: str, i: int) -> Any:
    values = block.get(name)
    return values[i] if isinstance(values, list) and i < len(values) else None


async def fetch_forecast(
    session: aiohttp.ClientSession, lat: float, lon: float, tz_name: str, days: int = 7
) -> tuple[list[HourPoint], dict[date, dict]]:
    """Hourly points (raw temperature only — correction is the caller's) + daily extras."""
    data = await _get(
        session,
        FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": HOURLY,
            "daily": DAILY,
            "forecast_days": days,
            "timezone": tz_name,
            "timeformat": "unixtime",
            "wind_speed_unit": "kmh",
        },
    )
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        raise OpenMeteoError("no hourly forecast")
    points: list[HourPoint] = []
    for i, ts in enumerate(times):
        code = _col(hourly, "weather_code", i)
        points.append(
            HourPoint(
                time=_utc(ts),
                temperature=None,
                raw_temperature=_col(hourly, "temperature_2m", i),
                humidity=_col(hourly, "relative_humidity_2m", i),
                dew_point=_col(hourly, "dew_point_2m", i),
                weather_code=int(code) if code is not None else None,
                is_day=bool(_col(hourly, "is_day", i) if _col(hourly, "is_day", i) is not None else 1),
                cloud_cover=_col(hourly, "cloud_cover", i),
                wind_speed=_col(hourly, "wind_speed_10m", i),
                wind_gust=_col(hourly, "wind_gusts_10m", i),
                wind_bearing=_col(hourly, "wind_direction_10m", i),
                precipitation=_col(hourly, "precipitation", i),
                precipitation_probability=_col(hourly, "precipitation_probability", i),
                pressure=_col(hourly, "pressure_msl", i),
                uv_index=_col(hourly, "uv_index", i),
            )
        )

    daily = data.get("daily") or {}
    offset = int(data.get("utc_offset_seconds") or 0)
    meta: dict[date, dict] = {}
    for i, ts in enumerate(daily.get("time") or []):
        # Daily times are local midnights expressed as unix time.
        day = datetime.fromtimestamp(ts + offset, tz=UTC).date()
        code = _col(daily, "weather_code", i)
        meta[day] = {
            "weather_code": int(code) if code is not None else None,
            "precipitation": _col(daily, "precipitation_sum", i),
            "precipitation_probability": _col(daily, "precipitation_probability_max", i),
            "wind_speed": _col(daily, "wind_speed_10m_max", i),
            "wind_gust": _col(daily, "wind_gusts_10m_max", i),
            "wind_bearing": _col(daily, "wind_direction_10m_dominant", i),
            "uv_index": _col(daily, "uv_index_max", i),
        }
    return points, meta


async def fetch_past_forecasts(
    session: aiohttp.ClientSession, lat: float, lon: float, past_days: int
) -> dict[datetime, float]:
    """What the model forecast one day ahead for each past hour (UTC → °C)."""
    data = await _get(
        session,
        PREVIOUS_RUNS_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m_previous_day1",
            "past_days": past_days,
            "forecast_days": 1,
            "timeformat": "unixtime",
        },
    )
    hourly = data.get("hourly") or {}
    out: dict[datetime, float] = {}
    for ts, value in zip(hourly.get("time") or [], hourly.get("temperature_2m_previous_day1") or [], strict=False):
        if value is not None:
            out[_utc(ts)] = float(value)
    return out
