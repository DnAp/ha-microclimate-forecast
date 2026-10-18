"""Pure forecast-correction model — no Home Assistant imports, unit-testable alone.

The idea: a weather model forecasts for a grid cell, a station measures one spot.
The difference between them is not random — at a given spot it repeats by time of
day (a yard where cold air pools at night, a sensor that the morning sun reaches)
and by weather (a clear calm night pools more cold air than a cloudy windy one).

Two models, both learned from what the station measured vs what the model had
forecast for the same hours one day ahead:

* **Analogs** (primary). The forecast is cut into nights (18:00–08:00) and days
  (09:00–17:00). For each coming segment we find the ANALOG_NEIGHBOURS past
  segments whose forecast looked most alike — temperature profile, cloud, wind,
  humidity — and add their weighted mean error, hour by hour.
* **Per-hour bias** (fallback). The mean error per local hour. Used where there
  are not enough past segments yet, and shown as the station's signature.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, tzinfo

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


# ── Analogs ────────────────────────────────────────────────────────────────

# Segment hours (local, on-the-hour instants). Together they cover all 24.
SEGMENT_HOURS: dict[str, list[int]] = {
    "night": [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    "day": [9, 10, 11, 12, 13, 14, 15, 16, 17],
}
# Backtested on 150 nights and days (each predicted from the 45 days before it):
# night low error 1.00° → 0.88°, day high 0.96° → 0.87° vs the per-hour bias. Fewer
# neighbours is noisy; more pulls in nights that did not look alike.
ANALOG_NEIGHBOURS = 8
# Scales that make a feature difference of "1" comparable: 2 °C of forecast
# temperature at an hour ≈ 40 % cloud ≈ 5 km/h wind ≈ 15 % humidity.
SCALE_TEMP = 2.0
SCALE_CLOUD = 40.0
SCALE_WIND = 5.0
SCALE_HUMIDITY = 15.0
# Keeps one almost identical analog from taking all the weight.
DISTANCE_FLOOR = 0.05
# A past segment with fewer measured instants than this share is not an analog.
MIN_SEGMENT_COVERAGE = 0.8


@dataclass
class PastHour:
    """What the model forecast one day ahead for an instant."""

    temperature: float
    cloud_cover: float | None = None
    wind_speed: float | None = None
    humidity: float | None = None


@dataclass
class Segment:
    kind: str  # "night" | "day"
    anchor: date  # local date the segment starts on
    temps: list[float | None]  # forecast at each instant of SEGMENT_HOURS[kind]
    cloud: float | None = None
    wind: float | None = None
    humidity: float | None = None
    errors: list[float | None] | None = None  # measured − forecast; past segments only

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "anchor": self.anchor.isoformat(),
            "temps": self.temps,
            "cloud": self.cloud,
            "wind": self.wind,
            "humidity": self.humidity,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Segment:
        return cls(
            kind=d["kind"],
            anchor=date.fromisoformat(d["anchor"]),
            temps=d["temps"],
            cloud=d.get("cloud"),
            wind=d.get("wind"),
            humidity=d.get("humidity"),
            errors=d.get("errors"),
        )


def segment_of(local: datetime) -> tuple[str, date]:
    """Which segment an instant belongs to: (kind, local start date)."""
    if 9 <= local.hour < 18:
        return "day", local.date()
    if local.hour >= 18:
        return "night", local.date()
    return "night", local.date() - timedelta(days=1)


def segment_instants(kind: str, anchor: date, tz: tzinfo) -> list[datetime]:
    """UTC instants of a segment, in SEGMENT_HOURS order."""
    out = []
    for h in SEGMENT_HOURS[kind]:
        day = anchor + timedelta(days=1) if kind == "night" and h < 18 else anchor
        out.append(datetime(day.year, day.month, day.day, h, tzinfo=tz).astimezone(UTC))
    return out


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def build_past_segments(observed: dict[datetime, float], past: dict[datetime, PastHour], tz: tzinfo) -> list[Segment]:
    """Past segments with their errors.

    `observed` holds hourly means keyed by the UTC start of the hour, so the station
    at an instant is the mean of the two hours around it.
    """
    if not past:
        return []
    first = min(past).astimezone(tz).date() - timedelta(days=1)
    last = max(past).astimezone(tz).date()
    out: list[Segment] = []
    day = first
    while day <= last:
        for kind in SEGMENT_HOURS:
            instants = segment_instants(kind, day, tz)
            hours = [past.get(t) for t in instants]
            if any(h is None for h in hours):
                continue
            errors: list[float | None] = []
            for t, h in zip(instants, hours, strict=True):
                a, b = observed.get(t - timedelta(hours=1)), observed.get(t)
                errors.append(None if a is None or b is None else round((a + b) / 2 - h.temperature, 2))  # type: ignore[union-attr]
            if sum(e is not None for e in errors) < MIN_SEGMENT_COVERAGE * len(errors):
                continue
            out.append(
                Segment(
                    kind=kind,
                    anchor=day,
                    temps=[h.temperature for h in hours],  # type: ignore[union-attr]
                    cloud=_mean([h.cloud_cover for h in hours]),  # type: ignore[union-attr]
                    wind=_mean([h.wind_speed for h in hours]),  # type: ignore[union-attr]
                    humidity=_mean([h.humidity for h in hours]),  # type: ignore[union-attr]
                    errors=errors,
                )
            )
        day += timedelta(days=1)
    return out


def segment_distance(a: Segment, b: Segment) -> float | None:
    """RMS feature difference over what both segments have. None if no overlap."""
    terms = [
        ((x - y) / SCALE_TEMP) ** 2 for x, y in zip(a.temps, b.temps, strict=True) if x is not None and y is not None
    ]
    if not terms:
        return None
    for x, y, scale in (
        (a.cloud, b.cloud, SCALE_CLOUD),
        (a.wind, b.wind, SCALE_WIND),
        (a.humidity, b.humidity, SCALE_HUMIDITY),
    ):
        if x is not None and y is not None:
            terms.append(((x - y) / scale) ** 2)
    return math.sqrt(sum(terms) / len(terms))


@dataclass
class AnalogModel:
    segments: list[Segment]
    learned_at: datetime
    days: int
    mae_raw: float | None = None
    mae_corrected: float | None = None

    def correction(self, target: Segment) -> list[float | None] | None:
        """Per-instant correction for a segment, or None without enough analogs."""
        scored = []
        for s in self.segments:
            if s.kind != target.kind or s.anchor == target.anchor or s.errors is None:
                continue
            d = segment_distance(target, s)
            if d is not None:
                scored.append((d, s))
        if len(scored) < ANALOG_NEIGHBOURS:
            return None
        scored.sort(key=lambda x: x[0])
        nearest = scored[:ANALOG_NEIGHBOURS]
        out: list[float | None] = []
        for i in range(len(target.temps)):
            num = den = 0.0
            for d, s in nearest:
                e = s.errors[i]  # type: ignore[index]
                if e is None:
                    continue
                w = 1.0 / (d + DISTANCE_FLOOR)
                num += w * e
                den += w
            out.append(round(num / den, 2) if den else None)
        return out

    def as_dict(self) -> dict:
        return {
            "segments": [s.as_dict() for s in self.segments],
            "learned_at": self.learned_at.isoformat(),
            "days": self.days,
            "mae_raw": self.mae_raw,
            "mae_corrected": self.mae_corrected,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnalogModel:
        return cls(
            segments=[Segment.from_dict(s) for s in d["segments"]],
            learned_at=datetime.fromisoformat(d["learned_at"]),
            days=int(d["days"]),
            mae_raw=d.get("mae_raw"),
            mae_corrected=d.get("mae_corrected"),
        )


def learn_analogs(segments: list[Segment], now: datetime) -> AnalogModel | None:
    """Keep the past segments; backtest each one against analogs from the others."""
    if not segments:
        return None
    model = AnalogModel(segments=segments, learned_at=now, days=len({s.anchor for s in segments}))
    raw = cor = 0.0
    n = 0
    for s in segments:
        corr = model.correction(s)
        if corr is None:
            continue
        for e, c in zip(s.errors or [], corr, strict=False):
            if e is None or c is None:
                continue
            raw += abs(e)
            cor += abs(e - c)
            n += 1
    if n:
        model.mae_raw = round(raw / n, 2)
        model.mae_corrected = round(cor / n, 2)
    return model


def target_segments(points: list[HourPoint], tz: tzinfo) -> dict[tuple[str, date], Segment]:
    """Forecast segments built from the coming hours (a running one is partial)."""
    by_key: dict[tuple[str, date], dict[datetime, HourPoint]] = defaultdict(dict)
    for p in points:
        by_key[segment_of(p.time.astimezone(tz))][p.time] = p
    out: dict[tuple[str, date], Segment] = {}
    for (kind, anchor), have in by_key.items():
        pts = [have.get(t) for t in segment_instants(kind, anchor, tz)]
        out[(kind, anchor)] = Segment(
            kind=kind,
            anchor=anchor,
            temps=[p.raw_temperature if p else None for p in pts],
            cloud=_mean([p.cloud_cover for p in pts if p]),
            wind=_mean([p.wind_speed for p in pts if p]),
            humidity=_mean([p.humidity for p in pts if p]),
        )
    return out


def apply_corrections(points: list[HourPoint], analogs: AnalogModel | None, bias: BiasModel | None, tz: tzinfo) -> int:
    """Corrected temperature for every point: analogs where they exist, else the
    per-hour bias, else the raw forecast. Returns how many points used analogs."""
    apply_bias(points, bias, tz)
    if analogs is None:
        return 0
    used = 0
    for (kind, anchor), seg in target_segments(points, tz).items():
        corr = analogs.correction(seg)
        if corr is None:
            continue
        index = {t: i for i, t in enumerate(segment_instants(kind, anchor, tz))}
        for p in points:
            i = index.get(p.time)
            if i is None or p.raw_temperature is None or corr[i] is None:
                continue
            p.temperature = round(p.raw_temperature + corr[i], 1)  # type: ignore[operator]
            used += 1
    return used


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
