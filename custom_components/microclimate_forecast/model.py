"""Pure forecast-correction model — no Home Assistant imports, unit-testable alone.

The idea: a weather model forecasts for a grid cell, a station measures one spot.
The difference between them is not random — at a given spot it repeats by time of
day (a yard where cold air pools at night, a sensor that the morning sun reaches).
So we pair the station's hourly means with what the model had forecast for those
same hours one day ahead, learn the mean error per local hour, and add it to the
current forecast.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo

# Weight of the neighbouring hours when learning one hour's bias (centre weighs 2).
NEIGHBOUR_WEIGHT = 1.0
CENTRE_WEIGHT = 2.0
# Pulls a bias learned from few samples towards zero: acts like this many extra
# zero-error samples (in weighted units — one day adds 4 per hour). Thirty days
# keep 98% of the learned bias; a two-day-old install keeps 80%.
SHRINK_SAMPLES = 2.0
# An hour with fewer samples than this (before smoothing) is not trusted at all.
MIN_SAMPLES_PER_HOUR = 3


@dataclass
class ErrorSample:
    """One hour: what the station measured minus what the model forecast for it."""

    day: date  # local date — the unit left out in the backtest
    hour: int  # local hour of the hour's midpoint, 0..23
    error: float


@dataclass
class BiasModel:
    """Learned per-local-hour correction (°C to add to the forecast)."""

    hours: list[float]
    samples: list[int]
    days: int
    learned_at: datetime
    # Backtest, each day predicted by a model learned without that day.
    mae_raw: float | None = None
    mae_corrected: float | None = None

    def bias_at(self, local: datetime) -> float:
        """Correction for an instant, interpolated between the hour midpoints."""
        x = local.hour + local.minute / 60 - 0.5
        i = math.floor(x)
        frac = x - i
        a = self.hours[i % 24]
        b = self.hours[(i + 1) % 24]
        return a + (b - a) * frac

    def as_dict(self) -> dict:
        return {
            "hours": self.hours,
            "samples": self.samples,
            "days": self.days,
            "learned_at": self.learned_at.isoformat(),
            "mae_raw": self.mae_raw,
            "mae_corrected": self.mae_corrected,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BiasModel:
        return cls(
            hours=[float(v) for v in d["hours"]],
            samples=[int(v) for v in d["samples"]],
            days=int(d["days"]),
            learned_at=datetime.fromisoformat(d["learned_at"]),
            mae_raw=d.get("mae_raw"),
            mae_corrected=d.get("mae_corrected"),
        )


def pair_samples(
    observed: dict[datetime, float],
    forecast: dict[datetime, float],
    tz: tzinfo,
) -> list[ErrorSample]:
    """Pair hourly station means with the past forecast for the same hours.

    `observed` is keyed by the UTC start of the hour and holds the mean over that
    hour (recorder statistics). `forecast` is keyed by UTC on-the-hour instants and
    holds instantaneous values, so the hour's forecast is the mean of its two ends.
    """
    out: list[ErrorSample] = []
    for start, obs in observed.items():
        f0 = forecast.get(start)
        f1 = forecast.get(start + timedelta(hours=1))
        if f0 is None or f1 is None or obs is None:
            continue
        mid = (start + timedelta(minutes=30)).astimezone(tz)
        out.append(ErrorSample(day=mid.date(), hour=mid.hour, error=obs - (f0 + f1) / 2))
    return out


def _fit(samples: list[ErrorSample]) -> tuple[list[float], list[int]]:
    sums = [0.0] * 24
    counts = [0] * 24
    for s in samples:
        sums[s.hour] += s.error
        counts[s.hour] += 1
    hours: list[float] = []
    for h in range(24):
        if counts[h] < MIN_SAMPLES_PER_HOUR:
            hours.append(0.0)
            continue
        num = 0.0
        den = 0.0
        for off, w in ((-1, NEIGHBOUR_WEIGHT), (0, CENTRE_WEIGHT), (1, NEIGHBOUR_WEIGHT)):
            k = (h + off) % 24
            num += w * sums[k]
            den += w * counts[k]
        hours.append(num / (den + SHRINK_SAMPLES))
    return hours, counts


def learn_bias(samples: list[ErrorSample], now: datetime) -> BiasModel | None:
    """Fit the per-hour bias and backtest it leave-one-day-out. None without data."""
    if not samples:
        return None
    hours, counts = _fit(samples)
    by_day: dict[date, list[ErrorSample]] = defaultdict(list)
    for s in samples:
        by_day[s.day].append(s)

    raw_err = 0.0
    cor_err = 0.0
    n = 0
    if len(by_day) >= 2:
        for day, todays in by_day.items():
            others = [s for s in samples if s.day != day]
            fitted, _ = _fit(others)
            for s in todays:
                raw_err += abs(s.error)
                cor_err += abs(s.error - fitted[s.hour])
                n += 1
    return BiasModel(
        hours=[round(v, 2) for v in hours],
        samples=counts,
        days=len(by_day),
        learned_at=now,
        mae_raw=round(raw_err / n, 2) if n else None,
        mae_corrected=round(cor_err / n, 2) if n else None,
    )


# ── Forecast shaping ───────────────────────────────────────────────────────


@dataclass
class HourPoint:
    time: datetime  # UTC
    temperature: float | None
    raw_temperature: float | None
    humidity: float | None = None
    dew_point: float | None = None
    weather_code: int | None = None
    is_day: bool = True
    cloud_cover: float | None = None
    wind_speed: float | None = None
    wind_gust: float | None = None
    wind_bearing: float | None = None
    precipitation: float | None = None
    precipitation_probability: float | None = None
    pressure: float | None = None
    uv_index: float | None = None


@dataclass
class DayPoint:
    day: date
    temperature: float
    templow: float
    weather_code: int | None = None
    precipitation: float | None = None
    precipitation_probability: float | None = None
    wind_speed: float | None = None
    wind_gust: float | None = None
    wind_bearing: float | None = None
    uv_index: float | None = None


@dataclass
class ForecastData:
    hourly: list[HourPoint]
    daily: list[DayPoint]
    bias: BiasModel | None
    fetched_at: datetime
    extra: dict = field(default_factory=dict)


def apply_bias(points: list[HourPoint], bias: BiasModel | None, tz: tzinfo) -> None:
    """Set each point's corrected temperature from its raw one, in place."""
    for p in points:
        if p.raw_temperature is None:
            p.temperature = None
        elif bias is None:
            p.temperature = p.raw_temperature
        else:
            p.temperature = round(p.raw_temperature + bias.bias_at(p.time.astimezone(tz)), 1)


# How fast the station's current deviation from the corrected forecast fades out.
# Backtested on 30 evenings: the 1-hour error falls from 0.97° to 0.39°, the
# 6-hour one from 1.07° to 0.95°; beyond ~12 h a longer memory starts to hurt.
NOWCAST_TAU_HOURS = 4.0


def forecast_at(points: list[HourPoint], when: datetime) -> float | None:
    """Corrected forecast at an instant, linear between the hourly points."""
    before = [p for p in points if p.time <= when and p.temperature is not None]
    after = [p for p in points if p.time > when and p.temperature is not None]
    if not before or not after:
        return None
    a, b = before[-1], after[0]
    span = (b.time - a.time).total_seconds()
    frac = (when - a.time).total_seconds() / span if span else 0.0
    return a.temperature + (b.temperature - a.temperature) * frac  # type: ignore[operator]


def apply_nowcast(
    points: list[HourPoint], observed_now: float | None, now: datetime, tau_hours: float = NOWCAST_TAU_HOURS
) -> float | None:
    """Carry the station's current deviation into the next hours, fading out.

    Returns the deviation applied (°C), or None when there was nothing to anchor to.
    """
    if observed_now is None:
        return None
    expected = forecast_at(points, now)
    if expected is None:
        return None
    offset = observed_now - expected
    for p in points:
        if p.time <= now or p.temperature is None:
            continue
        hours = (p.time - now).total_seconds() / 3600
        p.temperature = round(p.temperature + offset * math.exp(-hours / tau_hours), 1)
    return round(offset, 2)


def daily_from_hourly(
    points: list[HourPoint], daily_meta: dict[date, dict], tz: tzinfo, min_hours: int = 20
) -> list[DayPoint]:
    """Daily high/low from the corrected hours; the rest from the model's daily values."""
    temps: dict[date, list[float]] = defaultdict(list)
    for p in points:
        if p.temperature is not None:
            temps[p.time.astimezone(tz).date()].append(p.temperature)
    out: list[DayPoint] = []
    for day in sorted(temps):
        values = temps[day]
        if len(values) < min_hours:
            continue
        meta = daily_meta.get(day, {})
        out.append(DayPoint(day=day, temperature=max(values), templow=min(values), **meta))
    return out


def window_extreme(
    points: list[HourPoint],
    now: datetime,
    tz: tzinfo,
    start_hour: int,
    end_hour: int,
    pick: str,
) -> tuple[float, datetime] | None:
    """Min or max of the corrected forecast over the current-or-next local window.

    The window runs from `start_hour` to `end_hour` local time and may cross
    midnight (night: 18 → 10). If `now` is inside a window, it is that window from
    now on; otherwise the next one.
    """
    local = now.astimezone(tz)
    crosses = end_hour <= start_hour

    def window_for(anchor: date) -> tuple[datetime, datetime]:
        start = datetime(anchor.year, anchor.month, anchor.day, start_hour, tzinfo=tz)
        end_day = anchor + timedelta(days=1) if crosses else anchor
        end = datetime(end_day.year, end_day.month, end_day.day, end_hour, tzinfo=tz)
        return start, end

    candidates = [
        window_for(local.date() - timedelta(days=1)),
        window_for(local.date()),
        window_for(local.date() + timedelta(days=1)),
    ]
    for start, end in candidates:
        if end <= local:
            continue
        lo = max(start, local)
        chosen = [p for p in points if p.temperature is not None and lo <= p.time.astimezone(tz) <= end]
        if not chosen:
            continue
        best = min(chosen, key=lambda p: p.temperature) if pick == "min" else max(chosen, key=lambda p: p.temperature)
        return best.temperature, best.time  # type: ignore[return-value]
    return None


# WMO weather interpretation codes → Home Assistant conditions.
WMO_CONDITIONS: dict[int, str] = {
    0: "sunny",
    1: "sunny",
    2: "partlycloudy",
    3: "cloudy",
    45: "fog",
    48: "fog",
    51: "rainy",
    53: "rainy",
    55: "rainy",
    56: "snowy-rainy",
    57: "snowy-rainy",
    61: "rainy",
    63: "rainy",
    65: "pouring",
    66: "snowy-rainy",
    67: "snowy-rainy",
    71: "snowy",
    73: "snowy",
    75: "snowy",
    77: "snowy",
    80: "rainy",
    81: "rainy",
    82: "pouring",
    85: "snowy",
    86: "snowy",
    95: "lightning-rainy",
    96: "lightning-rainy",
    99: "lightning-rainy",
}


def condition_for(code: int | None, is_day: bool = True) -> str | None:
    if code is None:
        return None
    cond = WMO_CONDITIONS.get(code)
    if cond == "sunny" and not is_day:
        return "clear-night"
    return cond
