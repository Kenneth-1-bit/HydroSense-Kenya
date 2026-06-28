import unittest
import numpy as np
import pandas as pd

from src.simulation import (
    simulate_euler,
    simulate_rk4
)

from src.optimization import (
    schedule_zone,
    schedule_all_zones,
    efficiency_report
)


class TestIntegration(unittest.TestCase):

    def setUp(self):

        self.days = 14

        self.rainfall = np.array([
            5, 0, 3, 0, 0, 10, 2,
            0, 0, 6, 0, 1, 4, 0
        ], dtype=float)

        self.temp = np.full(self.days, 25.0)
        self.wind = np.full(self.days, 2.0)
        self.solar = np.full(self.days, 0.8)
        self.humidity = np.full(self.days, 70.0)

        self.irrigation = np.zeros(self.days)

        self.crop = pd.DataFrame({

            "zone_id": [
                "Zone_A",
                "Zone_B",
                "Zone_C"
            ],

            "min_moisture_pct": [
                25,
                28,
                22
            ],

            "target_moisture_pct": [
                35,
                38,
                32
            ],

            "field_capacity_pct": [
                45,
                48,
                42
            ],

            "drainage_coefficient": [
                0.20,
                0.25,
                0.15
            ]
        })

        self.initial = {
            "Zone_A": 30,
            "Zone_B": 32,
            "Zone_C": 28
        }

    # ==========================================================
    # COMPLETE PIPELINE
    # ==========================================================

    def test_complete_pipeline(self):

        # ----------------------------
        # Euler Simulation
        # ----------------------------

        euler = simulate_euler(

            initial=30,

            rainfall=self.rainfall,

            irrigation=self.irrigation,

            temp=self.temp,

            wind=self.wind,

            solar=self.solar,

            humidity=self.humidity,

            field_capacity=45,

            drainage_coeff=0.20

        )

        self.assertEqual(
            len(euler),
            self.days + 1
        )

        # ----------------------------
        # RK4 Simulation
        # ----------------------------

        rk4 = simulate_rk4(

            initial=30,

            rainfall=self.rainfall,

            irrigation=self.irrigation,

            temp=self.temp,

            wind=self.wind,

            solar=self.solar,

            humidity=self.humidity,

            field_capacity=45,

            drainage_coeff=0.20

        )

        self.assertEqual(
            len(rk4),
            self.days + 1
        )

        # ----------------------------
        # Single-zone optimisation
        # ----------------------------

        irrigation_schedule, moisture = schedule_zone(

            S0=30,

            rainfall=self.rainfall,

            temp=self.temp,

            wind=self.wind,

            solar=self.solar,

            humidity=self.humidity,

            min_moisture=25,

            target_moisture=35,

            field_capacity=45,

            drainage_coeff=0.20,

            efficiency=0.90

        )

        self.assertEqual(
            len(irrigation_schedule),
            self.days
        )

        self.assertEqual(
            len(moisture),
            self.days + 1
        )

        # ----------------------------
        # Multi-zone optimisation
        # ----------------------------

        schedules, moistures = schedule_all_zones(

            S0_dict=self.initial,

            rainfall=self.rainfall,

            temp=self.temp,

            wind=self.wind,

            solar=self.solar,

            humidity=self.humidity,

            params_df=self.crop,

            efficiency=0.90

        )

        self.assertEqual(
            len(schedules),
            3
        )

        self.assertEqual(
            len(moistures),
            3
        )

        # ----------------------------
        # Efficiency report
        # ----------------------------

        report = pd.DataFrame(

            efficiency_report(

                schedules,

                moistures,

                self.crop

            )

        )

        self.assertEqual(
            len(report),
            3
        )

        expected_columns = {

            "zone",

            "total_irrigation_mm",

            "days_irrigated",

            "stress_days",

            "mean_moisture_pct",

            "max_moisture_pct",

            "min_moisture_pct"

        }

        self.assertTrue(
            expected_columns.issubset(report.columns)
        )

        # ----------------------------
        # Sanity checks
        # ----------------------------

        self.assertTrue(
            np.all(report["total_irrigation_mm"] >= 0)
        )

        self.assertTrue(
            np.all(report["days_irrigated"] >= 0)
        )

        self.assertTrue(
            np.all(report["stress_days"] >= 0)
        )

        self.assertTrue(
            np.all(np.isfinite(euler))
        )

        self.assertTrue(
            np.all(np.isfinite(rk4))
        )

        self.assertTrue(
            np.all(np.isfinite(moisture))
        )


if __name__ == "__main__":
    unittest.main()
