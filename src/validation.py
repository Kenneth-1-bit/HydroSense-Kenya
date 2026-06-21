import numpy as np
import pandas as pd


def require_columns(df, required):

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"DataFrame is missing required column(s): {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


def check_range(df, column, low, high):

    require_columns(df, [column])
    mask = (df[column] < low) | (df[column] > high)
    return df[mask].copy()


def validate_arrays(*arrays, names=None):

    if names is None:
        names = [f"array_{i}" for i in range(len(arrays))]

    arrays = [np.asarray(a, dtype=float) for a in arrays]
    lengths = {name: len(arr) for name, arr in zip(names, arrays)}

    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        raise ValueError(
            f"All simulation arrays must have the same length. "
            f"Got: {lengths}"
        )

    for name, arr in zip(names, arrays):
        n_nan = np.isnan(arr).sum()
        if n_nan:
            raise ValueError(
                f"Array '{name}' contains {n_nan} NaN value(s). "
                "Fill or drop missing values before simulation."
            )


def validate_moisture(value, label="moisture"):

    if not (0.0 <= value <= 100.0):
        raise ValueError(
            f"'{label}' = {value} is outside the physical range [0, 100]%."
        )
