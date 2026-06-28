import unittest
import pandas as pd
import numpy as np

from src.preprocessing import (
    fix_dates,
    drop_duplicates,
    convert_numeric,
    fill_rainfall,
    range_filter,
    remove_outliers
)


class TestPreprocessing(unittest.TestCase):

    # ---------------------------------------------------------
    # fix_dates()
    # ---------------------------------------------------------

    def test_fix_dates_removes_invalid_dates(self):

        df = pd.DataFrame({
            "date": [
                "2025-03-01",
                "invalid_date",
                "2025-03-03"
            ]
        })

        cleaned = fix_dates(df, "date")

        self.assertEqual(len(cleaned), 2)
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(
                cleaned["date"]
            )
        )

    # ---------------------------------------------------------
    # drop_duplicates()
    # ---------------------------------------------------------

    def test_drop_duplicates(self):

        df = pd.DataFrame({
            "id": [1, 1, 2, 3],
            "value": [10, 10, 20, 30]
        })

        cleaned = drop_duplicates(df, ["id"])

        self.assertEqual(len(cleaned), 3)

    # ---------------------------------------------------------
    # convert_numeric()
    # ---------------------------------------------------------

    def test_convert_numeric(self):

        df = pd.DataFrame({
            "temperature": [
                "20",
                "25",
                "bad"
            ]
        })

        cleaned = convert_numeric(
            df,
            ["temperature"]
        )

        self.assertEqual(
            cleaned["temperature"].isna().sum(),
            1
        )

    # ---------------------------------------------------------
    # fill_rainfall()
    # ---------------------------------------------------------

    def test_fill_rainfall(self):

        df = pd.DataFrame({
            "rainfall_mm": [
                10,
                np.nan,
                20
            ]
        })

        cleaned = fill_rainfall(df)

        self.assertEqual(
            cleaned["rainfall_mm"].isna().sum(),
            0
        )

        self.assertAlmostEqual(
            cleaned.loc[1, "rainfall_mm"],
            15.0
        )

    def test_fill_rainfall_all_missing(self):

        df = pd.DataFrame({
            "rainfall_mm": [
                np.nan,
                np.nan
            ]
        })

        with self.assertRaises(ValueError):
            fill_rainfall(df)

    # ---------------------------------------------------------
    # range_filter()
    # ---------------------------------------------------------

    def test_range_filter(self):

        limits = {
            "temperature": (10, 40)
        }

        df = pd.DataFrame({
            "temperature": [
                25,
                45,
                18
            ]
        })

        cleaned = range_filter(df, limits)

        self.assertEqual(len(cleaned), 2)

    # ---------------------------------------------------------
    # remove_outliers()
    # ---------------------------------------------------------

    def test_remove_outliers(self):

        df = pd.DataFrame({
            "temperature": [
                22,
                23,
                24,
                25,
                150
            ]
        })

        cleaned = remove_outliers(
            df,
            ["temperature"]
        )

        self.assertEqual(len(cleaned), 4)


if __name__ == "__main__":
    unittest.main()
