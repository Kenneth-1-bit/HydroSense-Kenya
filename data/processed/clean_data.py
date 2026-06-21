from pathlib import Path
import pandas as pd
import numpy as np


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def fill_rainfall_mean(df):
    """Fill missing rainfall values using mean."""
    mean_rain = df["rainfall_mm"].mean()

    df["rainfall_mm"] = df["rainfall_mm"].fillna(
        round(mean_rain, 2)
    )

    return df


def remove_duplicates(df, subset_cols):
    """Drop duplicate records."""
    return df.drop_duplicates(
        subset=subset_cols,
        keep="first"
    )


def convert_numeric(df, columns):
    """
    Convert columns to numbers.
    Drop rows that cannot be converted.
    """

    for col in columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        df = df[df[col].notna()]

    return df


def fix_dates(df, column):
    """Convert dates to standard format."""

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )

    df = df[df[column].notna()]

    return df


# ------------------------------------------------------------------
# Unit fixes
# ------------------------------------------------------------------

def try_unit_conversion(value, column):

    # Fahrenheit -> Celsius
    if column == "temperature_c" and value > 50:

        celsius = (value - 32) * 5 / 9

        if 10 <= celsius <= 40:
            return round(celsius, 2)

    # cm -> mm
    if column == "rainfall_mm" and 100 < value <= 1000:

        mm = value / 10

        if mm <= 100:
            return round(mm, 2)

    # km/h -> m/s
    if column == "wind_speed_mps" and 15 < value < 54:

        mps = value / 3.6

        if mps <= 15:
            return round(mps, 2)

    # mL -> L
    if column == "tank_level_liters" and value > 6000:

        liters = value / 1000

        if 100 <= liters <= 6000:
            return round(liters, 2)

    return None


# ------------------------------------------------------------------
# Kenya realistic ranges
# ------------------------------------------------------------------

WEATHER_LIMITS = {
    "temperature_c": (10, 40),
    "humidity_pct": (30, 100),
    "rainfall_mm": (0, 100),
    "wind_speed_mps": (0, 15),
    "solar_index": (0, 1)
}

SOIL_LIMITS = {
    "soil_moisture_pct": (5, 60),
    "tank_level_liters": (100, 6000),
    "pump_flow_lpm": (5, 50),
    "pump_power_watts": (300, 700)
}


def apply_range_check(df, limits):

    rows_to_drop = []

    for col, (low, high) in limits.items():

        for idx, value in df[col].items():

            if pd.isna(value):
                continue

            if low <= value <= high:
                continue

            converted = try_unit_conversion(
                value,
                col
            )

            if converted is not None:

                df.at[idx, col] = converted

            else:

                rows_to_drop.append(idx)

    return df.drop(index=rows_to_drop)


# ------------------------------------------------------------------
# IQR outlier removal
# ------------------------------------------------------------------

def remove_iqr_outliers(df, columns):

    for col in columns:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df = df[
            (df[col] >= lower)
            &
            (df[col] <= upper)
        ]

    return df


# ------------------------------------------------------------------
# Weather dataset
# ------------------------------------------------------------------

def clean_weather():

    weather = pd.read_csv(
        RAW / "weather_daily.csv"
    )

    weather = fix_dates(
        weather,
        "date"
    )

    weather = remove_duplicates(
        weather,
        ["date"]
    )

    weather = convert_numeric(
        weather,
        [
            "rainfall_mm",
            "temperature_c",
            "humidity_pct",
            "wind_speed_mps",
            "solar_index"
        ]
    )

    weather = fill_rainfall_mean(weather)

    weather["humidity_pct"] = (
        weather["humidity_pct"]
        .fillna(weather["humidity_pct"].median())
    )

    weather = apply_range_check(
        weather,
        WEATHER_LIMITS
    )

    weather = remove_iqr_outliers(
        weather,
        [
            "temperature_c",
            "humidity_pct"
        ]
    )

    weather.to_csv(
        PROCESSED / "weather_daily_cleaned.csv",
        index=False
    )


# ------------------------------------------------------------------
# Soil dataset
# ------------------------------------------------------------------

def clean_soil():

    soil = pd.read_csv(
        RAW / "soil_sensor_data.csv"
    )

    soil = fix_dates(
        soil,
        "timestamp"
    )

    soil = remove_duplicates(
        soil,
        ["timestamp", "zone_id"]
    )

    soil = convert_numeric(
        soil,
        [
            "soil_moisture_pct",
            "tank_level_liters",
            "pump_flow_lpm",
            "pump_power_watts"
        ]
    )

    for col in [
        "soil_moisture_pct",
        "tank_level_liters",
        "pump_flow_lpm",
        "pump_power_watts"
    ]:

        zone_median = (
            soil.groupby("zone_id")[col]
            .transform("median")
        )

        soil[col] = soil[col].fillna(
            zone_median
        )

    soil = apply_range_check(
        soil,
        SOIL_LIMITS
    )

    soil = remove_iqr_outliers(
        soil,
        [
            "tank_level_liters",
            "pump_flow_lpm",
            "pump_power_watts"
        ]
    )

    soil.to_csv(
        PROCESSED / "soil_sensor_data_cleaned.csv",
        index=False
    )


# ------------------------------------------------------------------
# Crop parameters
# ------------------------------------------------------------------

def clean_crop_parameters():

    crop = pd.read_csv(
        RAW / "crop_zone_parameters.csv"
    )

    crop = remove_duplicates(
        crop,
        ["zone_id"]
    )

    crop.to_csv(
        PROCESSED /
        "crop_zone_parameters_cleaned.csv",
        index=False
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    
    clean_weather()
    print("Cleaning weather data...")

    clean_soil()
    print("Cleaning soil data...")

    clean_crop_parameters()
    print("Cleaning crop parameters...")

    print("Done.")


if __name__ == "__main__":
    main()
