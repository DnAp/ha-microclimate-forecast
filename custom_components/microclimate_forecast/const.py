"""Constants for Microclimate Forecast."""

from datetime import timedelta

DOMAIN = "microclimate_forecast"

CONF_TEMPERATURE = "temperature_entity"
CONF_HUMIDITY = "humidity_entity"
CONF_PRESSURE = "pressure_entity"
CONF_WIND_SPEED = "wind_speed_entity"
CONF_WIND_GUST = "wind_gust_entity"
CONF_WIND_BEARING = "wind_bearing_entity"
CONF_DEW_POINT = "dew_point_entity"
CONF_UV = "uv_index_entity"
CONF_LEARNING_DAYS = "learning_days"

OPTIONAL_SENSORS = (
    CONF_HUMIDITY,
    CONF_PRESSURE,
    CONF_WIND_SPEED,
    CONF_WIND_GUST,
    CONF_WIND_BEARING,
    CONF_DEW_POINT,
    CONF_UV,
)

DEFAULT_NAME = "Microclimate"
DEFAULT_LEARNING_DAYS = 45
MIN_LEARNING_DAYS = 7
MAX_LEARNING_DAYS = 90

UPDATE_INTERVAL = timedelta(hours=1)
RELEARN_INTERVAL = timedelta(hours=24)
FORECAST_DAYS = 7

# Local windows for the convenience sensors.
NIGHT_START_HOUR = 18
NIGHT_END_HOUR = 10
DAY_START_HOUR = 6
DAY_END_HOUR = 20

STORAGE_VERSION = 1
