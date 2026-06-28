import unittest
import numpy as np

from src.numerical_methods import (
    bisection,
    newton_raphson,
    secant,
    forward_difference,
    central_difference,
    trapezoidal,
    simpson,
    gaussian_elimination
)


class TestNumericalMethods(unittest.TestCase):

    # =========================================================
    # ROOT FINDING
    # =========================================================

    def test_bisection(self):

        f = lambda x: x**2 - 4

        result = bisection(f, 0, 3)

        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["root"], 2.0, places=5)

    def test_bisection_invalid_interval(self):

        f = lambda x: x**2 + 1

        with self.assertRaises(ValueError):
            bisection(f, -1, 1)

    def test_newton_raphson(self):

        f = lambda x: x**2 - 4
        df = lambda x: 2*x

        result = newton_raphson(f, df, 3)

        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["root"], 2.0, places=5)

    def test_newton_zero_derivative(self):

        f = lambda x: x**2
        df = lambda x: 2*x

        with self.assertRaises(ValueError):
            newton_raphson(f, df, 0)

    def test_secant(self):

        f = lambda x: x**2 - 4

        result = secant(f, 1, 3)

        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["root"], 2.0, places=5)

    def test_secant_same_initial_guess(self):

        f = lambda x: x**2 - 4

        with self.assertRaises(ValueError):
            secant(f, 2, 2)

    # =========================================================
    # NUMERICAL DIFFERENTIATION
    # =========================================================

    def test_forward_difference(self):

        f = lambda x: x**2

        derivative = forward_difference(f, 2)

        self.assertAlmostEqual(
            derivative,
            4,
            places=3
        )

    def test_central_difference(self):

        f = lambda x: x**2

        derivative = central_difference(f, 2)

        self.assertAlmostEqual(
            derivative,
            4,
            places=5
        )

    # =========================================================
    # NUMERICAL INTEGRATION
    # =========================================================

    def test_trapezoidal(self):

        x = np.linspace(0, 1, 101)
        y = x**2

        area = trapezoidal(x, y)

        self.assertAlmostEqual(
            area,
            1/3,
            places=3
        )

    def test_trapezoidal_invalid_lengths(self):

        with self.assertRaises(ValueError):

            trapezoidal(
                [0, 1],
                [1]
            )

    def test_simpson(self):

        x = np.linspace(0, 1, 101)
        y = x**2

        area = simpson(x, y)

        self.assertAlmostEqual(
            area,
            1/3,
            places=5
        )

    def test_simpson_too_few_points(self):

        with self.assertRaises(ValueError):

            simpson(
                [0, 1],
                [0, 1]
            )

    # =========================================================
    # LINEAR SYSTEMS
    # =========================================================

    def test_gaussian_elimination(self):

        A = np.array([
            [2, 1],
            [5, 7]
        ])

        b = np.array([
            11,
            13
        ])

        x = gaussian_elimination(A, b)

        expected = np.linalg.solve(A, b)

        np.testing.assert_allclose(
            x,
            expected,
            atol=1e-6
        )

    def test_gaussian_singular_matrix(self):

        A = np.array([
            [1, 2],
            [2, 4]
        ])

        b = np.array([
            1,
            2
        ])

        with self.assertRaises(ValueError):
            gaussian_elimination(A, b)


if __name__ == "__main__":
    unittest.main()
