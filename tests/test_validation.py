import unittest
import numpy as np
import pandas as pd

from src.validation import (
    require_columns,
    check_range,
    validate_arrays,
    validate_moisture
)


class TestValidation(unittest.TestCase):

    # ---------------------------------------------------------
    # require_columns()
    # ---------------------------------------------------------

    def test_require_columns_pass(self):

        df = pd.DataFrame({
            "temperature": [20],
            "humidity": [70]
        })

        # Should not raise an exception
        require_columns(
            df,
            ["temperature", "humidity"]
        )

    def test_require_columns_fail(self):

        df = pd.DataFrame({
            "temperature": [20]
        })

        with self.assertRaises(ValueError):
            require_columns(
                df,
                ["temperature", "humidity"]
            )

    # ---------------------------------------------------------
    # check_range()
    # ---------------------------------------------------------

    def test_check_range(self):

        df = pd.DataFrame({
            "temperature": [
                20,
                45,
                18,
                5
            ]
        })

        outliers = check_range(
            df,
            "temperature",
            10,
            40
        )

        self.assertEqual(len(outliers), 2)

        self.assertTrue(
            (outliers["temperature"] == [45, 5]).all()
        )

    # ---------------------------------------------------------
    # validate_arrays()
    # ---------------------------------------------------------

    def test_validate_arrays_pass(self):

        validate_arrays(
            [1, 2, 3],
            [4, 5, 6],
            names=["a", "b"]
        )

    def test_validate_arrays_different_lengths(self):

        with self.assertRaises(ValueError):

            validate_arrays(
                [1, 2, 3],
                [4, 5],
                names=["a", "b"]
            )

    def test_validate_arrays_nan(self):

        with self.assertRaises(ValueError):

            validate_arrays(
                [1, np.nan, 3],
                [4, 5, 6],
                names=["a", "b"]
            )

    # ---------------------------------------------------------
    # validate_moisture()
    # ---------------------------------------------------------

    def test_validate_moisture_valid(self):

        validate_moisture(45)

    def test_validate_moisture_negative(self):

        with self.assertRaises(ValueError):
            validate_moisture(-1)

    def test_validate_moisture_above_100(self):

        with self.assertRaises(ValueError):
            validate_moisture(120)


if __name__ == "__main__":
    unittest.main()
