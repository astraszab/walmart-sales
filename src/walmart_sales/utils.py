"""Utility funcitons."""

from dateutil.relativedelta import relativedelta
import pandas as pd


def get_week_diff(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Difference between start and end in whole weeks."""
    return int((end - start) / pd.Timedelta(1, "W"))


def add_weeks(date: pd.Timestamp, weeks: int) -> pd.Timestamp:
    """Add weeks to a date.

    Args:
        date: initial date.
        weeks: weeks to add.

    Returns:
        Date with added months.
    """
    return pd.Timestamp(date + relativedelta(weeks=weeks))
