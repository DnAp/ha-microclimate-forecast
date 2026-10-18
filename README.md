# Microclimate Forecast

A Home Assistant weather entity whose forecast matches **your own outdoor sensor**,
not the grid cell of a weather model.

A weather model forecasts for a few square kilometres; your sensor measures one
spot. At a given spot the difference is not random — it repeats by time of day: a
yard where cold air pools at night, a wall the afternoon sun reaches. This
integration learns that difference from your sensor's own history and adds it to
the [Open-Meteo](https://open-meteo.com/) forecast.

On the station it was built for (a GW1200A in a Cypriot yard) the raw forecast was
too warm **every single night** for a month, by 3–4 °C. After correction the mean
hourly error fell from **1.95 °C to 0.95 °C**.

## How it works

1. **Learning (once a day).** It reads the hourly means of your temperature sensor
   from the recorder's long-term statistics (falls back to state history for sensors
   without a `state_class`) and asks Open-Meteo what it had forecast *one day ahead*
   for each of those hours. The error is averaged per local hour (smoothed over
   neighbouring hours). Nothing leaves Home Assistant except the Open-Meteo requests.
2. **Forecast (every hour).** The current Open-Meteo forecast gets the learned
   correction for each hour of the day.
3. **Nowcast.** The sensor's deviation from the corrected forecast *right now* is
   carried into the next hours and fades out (half-life ≈ 3 h). In the backtest this
   cut the 1-hour error from 0.97 °C to 0.39 °C.
4. The learned correction is checked leave-one-day-out: each day is predicted by a
   model that never saw it. The result is exposed as `mae_raw` / `mae_corrected`.

Only temperature is corrected. Condition, precipitation, wind and the rest come
from Open-Meteo unchanged; *current* humidity, pressure, wind and UV come from your
sensors if you select them.

## Entities

| Entity | What |
|---|---|
| `weather.<name>` | Current readings from your sensors, hourly + daily forecast (corrected). Attributes: `bias_by_hour`, `mae_raw`, `mae_corrected`, `learning_days`, `learned_at`, `nowcast_offset`. |
| `sensor.<name>_night_low` | Lowest corrected temperature of the current-or-next night (18:00–10:00 local), attribute `at` (when), `uncorrected_forecast`. |
| `sensor.<name>_day_high` | Highest corrected temperature of the current-or-next day (06:00–20:00 local). |

## Install (HACS)

1. HACS → ⋮ → **Custom repositories** → add this repository's URL, type **Integration**.
2. Install **Microclimate Forecast**, restart Home Assistant.
3. Settings → Devices & services → **Add integration** → *Microclimate Forecast*.
4. Pick your outdoor temperature sensor (others are optional). Location is taken
   from Home Assistant's settings.

The correction needs history: with the recorder's long-term statistics it works from
the first minute (statistics are kept indefinitely). A sensor added yesterday gives
a weak, shrunk correction that improves day by day.

## Example: protect frost-sensitive plants

```yaml
automation:
  - alias: "Bring the dragon fruit in"
    triggers:
      - trigger: time
        at: "19:00:00"
    conditions:
      - condition: numeric_state
        entity_id: sensor.microclimate_night_low
        below: 5
    actions:
      - action: notify.mobile_app_phone
        data:
          message: >
            Tonight down to {{ states('sensor.microclimate_night_low') }} °C
            at {{ as_timestamp(state_attr('sensor.microclimate_night_low', 'at')) | timestamp_custom('%H:%M') }}
            — bring the young plants in.
```

## Data source

Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), including its
archive of previous model runs. Free for non-commercial use, no API key.
