"""The correction model on its own."""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from custom_components.microclimate_forecast.model import (
    BiasModel,
    HourPoint,
    apply_bias,
    apply_nowcast,
    condition_for,
    daily_from_hourly,
    learn_bias,
    pair_samples,
    window_extreme,
)

TZ = ZoneInfo("Asia/Nicosia")
START = datetime(2026, 9, 18, tzinfo=UTC)


def true_bias(local_hour: int) -> float:
    # Cold yard at night, sun on the sensor at noon — the shape seen at the real station.
    return -3.5 if local_hour < 8 or local_hour >= 21 else 0.7


def synthetic(days: int = 30, noise: float = 0.0):
    forecast, observed = {}, {}
    for h in range(days * 24 + 1):
        t = START + timedelta(hours=h)
        forecast[t] = 20.0 + (h % 7) * 0.3
    for h in range(days * 24):
        t = START + timedelta(hours=h)
        mid_local = (t + timedelta(minutes=30)).astimezone(TZ)
        f = (forecast[t] + forecast[t + timedelta(hours=1)]) / 2
        jitter = noise * (1 if (h * 7919) % 3 == 0 else -1)
        observed[t] = f + true_bias(mid_local.hour) + jitter
    return observed, forecast


def test_learns_the_station_shape():
    observed, forecast = synthetic()
    model = learn_bias(pair_samples(observed, forecast, TZ), START + timedelta(days=30))
    assert model is not None
    assert model.days >= 30
    # Away from the day/night edges the smoothed bias is the true one (slightly shrunk).
    assert abs(model.hours[3] - (-3.5)) < 0.15
    assert abs(model.hours[13] - 0.7) < 0.1
    assert model.mae_corrected < model.mae_raw / 3


def test_no_overlap_learns_nothing():
    observed, _ = synthetic(days=2)
    assert learn_bias(pair_samples(observed, {}, TZ), START) is None


def test_few_samples_are_not_trusted():
    observed, forecast = synthetic(days=1)
    only_two = dict(list(observed.items())[:2])
    model = learn_bias(pair_samples(only_two, forecast, TZ), START)
    assert model is not None and all(v == 0.0 for v in model.hours)


def test_bias_interpolates_between_hour_midpoints():
    m = BiasModel(hours=[0.0] * 24, samples=[10] * 24, days=1, learned_at=START)
    m.hours[3], m.hours[4] = -2.0, -4.0
    assert m.bias_at(datetime(2026, 1, 1, 3, 30, tzinfo=TZ)) == -2.0
    assert m.bias_at(datetime(2026, 1, 1, 4, 0, tzinfo=TZ)) == -3.0
    m.hours[23], m.hours[0] = 1.0, 3.0
    assert m.bias_at(datetime(2026, 1, 1, 0, 0, tzinfo=TZ)) == 2.0  # wraps midnight


def test_round_trip_storage():
    m = learn_bias(pair_samples(*synthetic(days=3), TZ), START)
    assert BiasModel.from_dict(m.as_dict()) == m


def points_for(day_count: int = 2):
    pts = []
    base = datetime(2026, 10, 18, 0, tzinfo=TZ).astimezone(UTC)
    for h in range(day_count * 24):
        t = base + timedelta(hours=h)
        local = t.astimezone(TZ).hour
        raw = 14.0 + (8.0 if 10 <= local <= 16 else 0.0)
        pts.append(HourPoint(time=t, temperature=None, raw_temperature=raw))
    return pts


def test_daily_and_windows_use_the_corrected_hours():
    pts = points_for()
    model = BiasModel(hours=[true_bias(h) for h in range(24)], samples=[100] * 24, days=30, learned_at=START)
    apply_bias(pts, model, TZ)
    days = daily_from_hourly(pts, {date(2026, 10, 18): {"weather_code": 2}}, TZ)
    assert [d.day for d in days] == [date(2026, 10, 18), date(2026, 10, 19)]
    assert days[0].templow == 10.5 and days[0].temperature == 22.7
    assert days[0].weather_code == 2
    now = datetime(2026, 10, 18, 23, 30, tzinfo=TZ)
    low = window_extreme(pts, now, TZ, 18, 10, "min")
    assert low is not None and low[0] == 10.5
    assert low[1].astimezone(TZ) > now


def test_without_bias_the_forecast_is_passed_through():
    pts = points_for(1)
    apply_bias(pts, None, TZ)
    assert all(p.temperature == p.raw_temperature for p in pts)


def test_conditions():
    assert condition_for(0, True) == "sunny"
    assert condition_for(0, False) == "clear-night"
    assert condition_for(95) == "lightning-rainy"
    assert condition_for(None) is None


def test_nowcast_carries_the_current_deviation_and_fades():
    pts = points_for(1)
    apply_bias(pts, None, TZ)
    now = pts[2].time + timedelta(minutes=30)  # forecast there: 14.0
    offset = apply_nowcast(pts, 12.0, now)
    assert offset == -2.0
    assert pts[1].temperature == 14.0  # the past is left alone
    assert pts[3].temperature == pytest.approx(14.0 - 2.0 * 0.882, abs=0.06)  # 0.5 h ahead
    assert pts[20].temperature == pytest.approx(14.0, abs=0.06)  # 17.5 h ahead: faded


def test_nowcast_without_a_reading_changes_nothing():
    pts = points_for(1)
    apply_bias(pts, None, TZ)
    assert apply_nowcast(pts, None, pts[2].time) is None
    assert all(p.temperature == p.raw_temperature for p in pts)
