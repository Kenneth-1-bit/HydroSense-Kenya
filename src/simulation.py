import numpy as np
from config import ET_COEFF


# ─────────────────────────────────────────────────────────────────────────────
# EVAPOTRANSPIRATION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_et(temperature, wind_speed, solar_index, humidity):

    et = (
        ET_COEFF["temperature"] * temperature
        + ET_COEFF["wind"]        * wind_speed
        + ET_COEFF["solar"]       * solar_index
        + ET_COEFF["humidity"]    * humidity   # negative coefficient
    )
    return max(0.0, et)


def et_vectorized(temp_arr, wind_arr, solar_arr, humidity_arr):

    t = np.asarray(temp_arr,    dtype=float)
    w = np.asarray(wind_arr,    dtype=float)
    s = np.asarray(solar_arr,   dtype=float)
    h = np.asarray(humidity_arr, dtype=float)

    et = (
        ET_COEFF["temperature"] * t
        + ET_COEFF["wind"]      * w
        + ET_COEFF["solar"]     * s
        + ET_COEFF["humidity"]  * h
    )
    return np.maximum(0.0, et)


# ─────────────────────────────────────────────────────────────────────────────
# WATER BALANCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_drainage(moisture, field_capacity, drainage_coeff):

    excess = max(0.0, moisture - field_capacity)
    return drainage_coeff * excess


def water_balance(moisture, rainfall, irrigation,
                  evapotranspiration, drainage):

    result = (
        moisture
        + rainfall
        + irrigation
        - evapotranspiration
        - drainage
    )
    return max(0.0, result)


# ─────────────────────────────────────────────────────────────────────────────
# EULER SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_euler(initial, rainfall, irrigation,
                   temp, wind, solar, humidity,
                   field_capacity, drainage_coeff,
                   dt=1.0):

    rainfall    = np.asarray(rainfall,    dtype=float)
    irrigation  = np.asarray(irrigation,  dtype=float)
    temp        = np.asarray(temp,        dtype=float)
    wind        = np.asarray(wind,        dtype=float)
    solar       = np.asarray(solar,       dtype=float)
    humidity    = np.asarray(humidity,    dtype=float)

    n = len(rainfall)
    lengths = {len(rainfall), len(irrigation), len(temp),
               len(wind), len(solar), len(humidity)}
    if len(lengths) > 1:
        raise ValueError(
            f"All input arrays must have the same length. Got lengths: {lengths}"
        )

    S = np.zeros(n + 1)
    S[0] = initial

    for i in range(n):
        et = calculate_et(temp[i], wind[i], solar[i], humidity[i])
        d  = compute_drainage(S[i], field_capacity, drainage_coeff)
        dS = rainfall[i] + irrigation[i] - et - d
        S[i + 1] = max(0.0, S[i] + dt * dS)

    return S


# ─────────────────────────────────────────────────────────────────────────────
# RK4 SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_rk4(initial, rainfall, irrigation,
                 temp, wind, solar, humidity,
                 field_capacity, drainage_coeff,
                 dt=1.0):

    rainfall   = np.asarray(rainfall,   dtype=float)
    irrigation = np.asarray(irrigation, dtype=float)
    temp       = np.asarray(temp,       dtype=float)
    wind       = np.asarray(wind,       dtype=float)
    solar      = np.asarray(solar,      dtype=float)
    humidity   = np.asarray(humidity,   dtype=float)

    n = len(rainfall)
    lengths = {len(rainfall), len(irrigation), len(temp),
               len(wind), len(solar), len(humidity)}
    if len(lengths) > 1:
        raise ValueError(
            f"All input arrays must have the same length. Got lengths: {lengths}"
        )

    def dS_dt(S, i):
        """Rate of change at time step i given current moisture S."""
        et = calculate_et(temp[i], wind[i], solar[i], humidity[i])
        d  = compute_drainage(S, field_capacity, drainage_coeff)
        return rainfall[i] + irrigation[i] - et - d

    S = np.zeros(n + 1)
    S[0] = initial

    for i in range(n):
        k1 = dS_dt(S[i],               i)
        k2 = dS_dt(S[i] + 0.5*dt*k1,  i)
        k3 = dS_dt(S[i] + 0.5*dt*k2,  i)
        k4 = dS_dt(S[i] + dt*k3,       i)

        S[i + 1] = max(0.0, S[i] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4))

    return S


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_rainfall(mean, std, n_days=30, n_scenarios=1000, seed=42):

    rng = np.random.default_rng(seed)
    samples = rng.normal(loc=mean, scale=std, size=(n_scenarios, n_days))
    return np.clip(samples, 0.0, None)
