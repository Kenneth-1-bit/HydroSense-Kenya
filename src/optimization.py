import numpy as np
from .simulation import calculate_et, compute_drainage


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-STEP HELPER
# ─────────────────────────────────────────────────────────────────────────────

def irrigation_needed(current, target, field_capacity):

    if current >= target:
        return 0.0
    return max(0.0, min(target - current, field_capacity - current))


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-ZONE SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def schedule_zone(S0, rainfall, temp, wind, solar, humidity,
                  min_moisture, target_moisture, field_capacity,
                  drainage_coeff, efficiency=1.0):

    rainfall = np.asarray(rainfall, dtype=float)
    n = len(rainfall)

    irrigation = np.zeros(n)
    moisture   = np.zeros(n + 1)
    moisture[0] = S0

    for i in range(n):
        S_now = moisture[i]
        et    = calculate_et(temp[i], wind[i], solar[i], humidity[i])
        d     = compute_drainage(S_now, field_capacity, drainage_coeff)

        # Predict moisture with zero irrigation
        S_no_irr = max(0.0, S_now + rainfall[i] - et - d)

        if S_no_irr < min_moisture:
            # How much to apply to reach target (gross, before efficiency loss)
            net_required = irrigation_needed(S_no_irr, target_moisture, field_capacity)

            gross_required = net_required / max(efficiency, 1e-6)

            irrigation[i] = gross_required

        else:

            irrigation[i] = 0.0

        # Recompute moisture with irrigation applied
        d_final = compute_drainage(S_now + rainfall[i] + irrigation[i] * efficiency - et,
                                   field_capacity, drainage_coeff)
        moisture[i + 1] = max(
            0.0,
            S_now + rainfall[i] + irrigation[i] * efficiency - et - d_final
        )

    return irrigation, moisture


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-ZONE SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def schedule_all_zones(S0_dict, rainfall, temp, wind, solar, humidity,
                       params_df, efficiency=1.0):

    params     = params_df.set_index("zone_id")
    schedules  = {}
    moistures  = {}

    for zone_id in params.index:
        row = params.loc[zone_id]
        irr, moist = schedule_zone(
            S0             = S0_dict[zone_id],
            rainfall       = rainfall,
            temp           = temp,
            wind           = wind,
            solar          = solar,
            humidity       = humidity,
            min_moisture   = row["min_moisture_pct"],
            target_moisture= row["target_moisture_pct"],
            field_capacity = row["field_capacity_pct"],
            drainage_coeff = row["drainage_coefficient"],
            efficiency     = efficiency,
        )
        schedules[zone_id] = irr
        moistures[zone_id] = moist

    return schedules, moistures


# ─────────────────────────────────────────────────────────────────────────────
# EFFICIENCY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def efficiency_report(schedules, moistures, params_df):

    params  = params_df.set_index("zone_id")
    report  = []

    for zone_id, irr in schedules.items():
        sm      = moistures[zone_id][1:]   # exclude day-0 initial value
        min_m   = params.loc[zone_id, "min_moisture_pct"]

        report.append({
            "zone":                zone_id,
            "total_irrigation_mm": round(float(irr.sum()), 2),
            "days_irrigated":      int(np.sum(irr > 0)),
            "stress_days":         int(np.sum(sm < min_m)),
            "mean_moisture_pct":   round(float(sm.mean()), 2),
            "max_moisture_pct":    round(float(sm.max()),  2),
            "min_moisture_pct":    round(float(sm.min()),  2),
        })

    return report
