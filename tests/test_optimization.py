import unittest
import numpy as np
import pandas as pd

from src.optimization import (
    irrigation_needed,
    schedule_zone,
    schedule_all_zones,
    efficiency_report
)


class TestOptimization(unittest.TestCase):

    def setUp(self):

        self.days = 10

        self.rainfall = np.zeros(self.days)

        self.temp = np.full(self.days, 25.0)

        self.wind = np.full(self.days, 2.0)

        self.solar = np.full(self.days, 0.8)

        self.humidity = np.full(self.days, 70.0)

        self.crop = pd.DataFrame({

            "zone_id": ["Zone_A", "Zone_B"],

            "min_moisture_pct": [25, 30],

            "target_moisture_pct": [35, 40],

            "field_capacity_pct": [45, 50],

            "drainage_coefficient": [0.20, 0.25]

        })

        self.initial_conditions = {

            "Zone_A": 30,

            "Zone_B": 35

        }

    # ==========================================================
    # irrigation_needed()
    # ==========================================================

    def test_irrigation_needed_when_required(self):

        amount = irrigation_needed(
            current=20,
            target=35,
            field_capacity=45
        )

        self.assertEqual(amount, 15)

    def test_irrigation_not_needed(self):

        amount = irrigation_needed(
            current=40,
            target=35,
            field_capacity=45
        )

        self.assertEqual(amount, 0)

    def test_irrigation_limited_by_field_capacity(self):

        amount = irrigation_needed(
            current=40,
            target=60,
            field_capacity=45
        )

        self.assertEqual(amount, 5)

    # ==========================================================
    # schedule_zone()
    # ==========================================================

    def test_schedule_zone_returns_correct_lengths(self):

        irrigation, moisture = schedule_zone(

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
            len(irrigation),
            self.days
        )

        self.assertEqual(
            len(moisture),
            self.days + 1
        )

    def test_schedule_zone_non_negative_irrigation(self):

        irrigation, _ = schedule_zone(

            S0=30,

            rainfall=self.rainfall,

            temp=self.temp,

            wind=self.wind,

            solar=self.solar,

            humidity=self.humidity,

            min_moisture=25,

            target_moisture=35,

            field_capacity=45,

            drainage_coeff=0.20

        )

        self.assertTrue(
            np.all(irrigation >= 0)
        )

    def test_schedule_zone_non_negative_moisture(self):

        _, moisture = schedule_zone(

            S0=30,

            rainfall=self.rainfall,

            temp=self.temp,

            wind=self.wind,

            solar=self.solar,

            humidity=self.humidity,

            min_moisture=25,

            target_moisture=35,

            field_capacity=45,

            drainage_coeff=0.20

        )

        self.assertTrue(
            np.all(moisture >= 0)
        )

    # ==========================================================
    # schedule_all_zones()
    # ==========================================================

    def test_schedule_all_zones(self):

        schedules, moistures = schedule_all_zones(

            S0_dict=self.initial_conditions,

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
            2
        )

        self.assertEqual(
            len(moistures),
            2
        )

        self.assertIn(
            "Zone_A",
            schedules
        )

        self.assertIn(
            "Zone_B",
            schedules
        )

    # ==========================================================
    # efficiency_report()
    # ==========================================================

    def test_efficiency_report(self):

        schedules, moistures = schedule_all_zones(

            S0_dict=self.initial_conditions,

            rainfall=self.rainfall,

            temp=self.temp,

            wind=self.wind,

            solar=self.solar,

            humidity=self.humidity,

            params_df=self.crop,

            efficiency=0.90

        )

        report = efficiency_report(

            schedules,

            moistures,

            self.crop

        )

        self.assertEqual(
            len(report),
            2
        )

        self.assertIn(
            "zone",
            report[0]
        )

        self.assertIn(
            "total_irrigation_mm",
            report[0]
        )

        self.assertIn(
            "days_irrigated",
            report[0]
        )

        self.assertIn(
            "stress_days",
            report[0]
        )

        self.assertIn(
            "mean_moisture_pct",
            report[0]
        )

        self.assertIn(
            "max_moisture_pct",
            report[0]
        )

        self.assertIn(
            "min_moisture_pct",
            report[0]
        )

    def test_efficiency_report_values(self):

        schedules, moistures = schedule_all_zones(

            S0_dict=self.initial_conditions,

            rainfall=self.rainfall,

            temp=self.temp,

            wind=self.wind,

            solar=self.solar,

            humidity=self.humidity,

            params_df=self.crop

        )

        report = pd.DataFrame(

            efficiency_report(

                schedules,

                moistures,

                self.crop

            )

        )

        self.assertTrue(
            np.all(report["total_irrigation_mm"] >= 0)
        )

        self.assertTrue(
            np.all(report["days_irrigated"] >= 0)
        )

        self.assertTrue(
            np.all(report["stress_days"] >= 0)
        )


if __name__ == "__main__":
    unittest.main()
