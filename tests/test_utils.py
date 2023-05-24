"""Tests for util functions."""


import pandas as pd

from walmart_sales.utils import get_week_diff


def test_get_week_diff() -> None:
    """It returns a whole number of weeks between specified dates."""
    assert (
        get_week_diff(pd.Timestamp("2012-03-23"), pd.Timestamp("2012-04-20"))
        == 4
    )
