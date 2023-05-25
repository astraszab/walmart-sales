"""Metrics."""

from typing import Any

import pandas as pd


def smape(target: "pd.Series[Any]", forecast: "pd.Series[Any]") -> float:
    """Symmetrical Mean Absolute Percentage Error.

    Args:
        target: actual values.
        forecast: forecasted values.

    Returns:
        SMAPE value in percents.
    """
    return float(
        100
        * (
            2
            * (forecast - target).abs()
            / (forecast.abs() + target.abs() + 1e-6)
        ).mean()
    )


def mape(target: "pd.Series[Any]", forecast: "pd.Series[Any]") -> float:
    """Mean Absolute Percentage Error.

    Args:
        target: actual values.
        forecast: forecasted values.

    Returns:
        MAPE value in percents.
    """
    return float(
        100 * ((forecast - target).abs() / (target.abs() + 1e-6)).mean()
    )


def wape(target: "pd.Series[Any]", forecast: "pd.Series[Any]") -> float:
    """Weighted Average Percentage Error.

    Args:
        target: actual values.
        forecast: forecasted values.

    Returns:
        WAPE value in percents.
    """
    return float(
        100 * (forecast - target).abs().sum() / (target.abs().sum() + 1e-6)
    )
