"""Utility funcitons."""

import pandas as pd


def get_week_diff(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Difference between start and end in whole weeks."""
    return int((end - start) / pd.Timedelta(1, "W"))
