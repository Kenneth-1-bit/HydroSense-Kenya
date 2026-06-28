import unittest
import numpy as np

from src.simulation import (
    calculate_et,
    et_vectorized,
    compute_drainage,
    water_balance,
    simulate_euler,
    simulate_rk4,
    monte_carlo_rainfall
)


class TestSimulation(unittest.TestCase):

    def setUp(self):

        self.days = 10

        self.rainfall = np.zeros(self.days)
        self.irrigation = np.zeros(self.days)

        self.temp = np.full(self.days, 25.0)
        self.wind = np.full(self.days, 2.0)
        self.solar = np.full(self.days, 0.8)
        self.humidity = np.full(self.days, 70.0)

    # =========================================================
    # EVAPOTRANSPIRATION
    # =========================================================

    def test_calculate_et_positive(self):

        et = calculate_et(
            25,
            2,
            0.8,
            70
        )

        self.assertGreaterEqual(et, 0.0)

    def test_calculate_et_clipped_to_zero(self):

        et = calculate_et(
            0,
            0,
            0,
            1000
        )

        self.assertEqual(et, 0.0)

    def test_et_vectorized(self):

        et = et_vectorized(
            self.temp,
            self.wind,
            self.solar,
            self.humidity
        )

        self.assertEqual(len(et), self.days)

        self.assertTrue(
            np.all(et >= 0)
        )

    # =========================================================
    # DRAINAGE
    # =========================================================

    def test_compute_drainage_above_capacity(self):

        d = compute_drainage(
            moisture=50,
            field_capacity=40,
            drainage_coeff=0.25
        )

        self.assertAlmostEqual(d, 2.5)

    def test_compute_drainage_below_capacity(self):

        d = compute_drainage(
            moisture=30,
            field_capacity=40,
            drainage_coeff=0.25
        )

        self.assertEqual(d, 0.0)

    # =========================================================
    # WATER BALANCE
    # =========================================================

    def test_water_balance(self):

        s = water_balance(
            moisture=30,
            rainfall=5,
            irrigation=3,
            evapotranspiration=2,
            drainage=1
        )

        self.assertEqual(s, 35)

    def test_water_balance_not_negative(self):

        s = water_balance(
            moisture=1,
            rainfall=0,
            irrigation=0,
            evapotranspiration=10,
            drainage=5
        )

        self.assertEqual(s, 0)

    # =========================================================
    # EULER
    # =========================================================

    def test_simulate_euler(self):

        result = simulate_euler(
            initial=30,
            rainfall=self.rainfall,
            irrigation=self.irrigation,
            temp=self.temp,
            wind=self.wind,
            solar=self.solar,
            humidity=self.humidity,
            field_capacity=40,
            drainage_coeff=0.2
        )

        self.assertEqual(
            len(result),
            self.days + 1
        )

        self.assertTrue(
            np.all(np.isfinite(result))
        )

    def test_simulate_euler_length_error(self):

        with self.assertRaises(ValueError):

            simulate_euler(
                initial=30,
                rainfall=np.zeros(10),
                irrigation=np.zeros(9),
                temp=np.zeros(10),
                wind=np.zeros(10),
                solar=np.zeros(10),
                humidity=np.zeros(10),
                field_capacity=40,
                drainage_coeff=0.2
            )

    # =========================================================
    # RK4
    # =========================================================

    def test_simulate_rk4(self):

        result = simulate_rk4(
            initial=30,
            rainfall=self.rainfall,
            irrigation=self.irrigation,
            temp=self.temp,
            wind=self.wind,
            solar=self.solar,
            humidity=self.humidity,
            field_capacity=40,
            drainage_coeff=0.2
        )

        self.assertEqual(
            len(result),
            self.days + 1
        )

        self.assertTrue(
            np.all(np.isfinite(result))
        )

    def test_simulate_rk4_length_error(self):

        with self.assertRaises(ValueError):

            simulate_rk4(
                initial=30,
                rainfall=np.zeros(10),
                irrigation=np.zeros(8),
                temp=np.zeros(10),
                wind=np.zeros(10),
                solar=np.zeros(10),
                humidity=np.zeros(10),
                field_capacity=40,
                drainage_coeff=0.2
            )

    # =========================================================
    # MONTE CARLO
    # =========================================================

    def test_monte_carlo_shape(self):

        samples = monte_carlo_rainfall(
            mean=8,
            std=3,
            n_days=30,
            n_scenarios=1000,
            seed=42
        )

        self.assertEqual(
            samples.shape,
            (1000, 30)
        )

    def test_monte_carlo_non_negative(self):

        samples = monte_carlo_rainfall(
            mean=5,
            std=10,
            n_days=20,
            n_scenarios=50,
            seed=42
        )

        self.assertTrue(
            np.all(samples >= 0)
        )

    def test_monte_carlo_reproducible(self):

        a = monte_carlo_rainfall(
            mean=10,
            std=2,
            seed=123
        )

        b = monte_carlo_rainfall(
            mean=10,
            std=2,
            seed=123
        )

        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
