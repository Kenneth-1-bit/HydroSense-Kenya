# ── Kenya realistic daily climate limits (March) ─────────────────────────────
# Sources: Kenya Meteorological Department, climatestotravel.com
# Rainfall cap set to 60 mm/day (extreme event threshold).
# 100 mm was too permissive — it allowed the 85 mm outlier through unchecked.
WEATHER_LIMITS = {
    "temperature_c":  (10,  40),
    "humidity_pct":   (30, 100),
    "rainfall_mm":    (0,   60),   # was 100 — tightened to realistic daily max
    "wind_speed_mps": (0,   15),
    "solar_index":    (0,    1),
}

# ── Soil sensor plausible ranges ──────────────────────────────────────────────
SOIL_LIMITS = {
    "soil_moisture_pct": (5,   60),
    "tank_level_liters": (100, 6000),
    "pump_flow_lpm":     (5,   50),
    "pump_power_watts":  (300, 700),
}

# ── ET formula coefficients (from project brief) ──────────────────────────────
# ET = max(0, C_T*T + C_W*W + C_S*Solar - C_H*H)
ET_COEFF = {
    "temperature":  0.12,
    "wind":         0.35,
    "solar":        2.40,
    "humidity":    -0.025,
}

# ── Crop zone identifiers ─────────────────────────────────────────────────────
ZONES = ["Zone_A", "Zone_B", "Zone_C"]

# ── Required columns per dataset ─────────────────────────────────────────────
WEATHER_COLS  = ["date", "rainfall_mm", "temperature_c",
                 "humidity_pct", "wind_speed_mps", "solar_index"]
SOIL_COLS     = ["timestamp", "zone_id", "soil_moisture_pct",
                 "tank_level_liters", "pump_flow_lpm", "pump_power_watts"]
CROP_COLS     = ["zone_id", "crop_type", "area_m2", "min_moisture_pct",
                 "target_moisture_pct", "field_capacity_pct", "drainage_coefficient"]
