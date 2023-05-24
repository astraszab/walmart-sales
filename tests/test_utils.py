"""Tests for util functions."""


import pandas as pd

from walmart_sales.utils import add_weeks, get_week_diff


def test_get_week_diff() -> None:
    """It returns a whole number of weeks between specified dates."""
    assert (
        get_week_diff(pd.Timestamp("2012-03-23"), pd.Timestamp("2012-04-20"))
        == 4
    )


def test_add_weeks() -> None:
    """It returns a timestamp that is n weeks later than provided."""
    assert add_weeks(pd.Timestamp("2012-03-23"), 4) == pd.Timestamp(
        "2012-04-20"
    )
