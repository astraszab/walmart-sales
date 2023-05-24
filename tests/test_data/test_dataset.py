"""Tests for WalmartDataset."""


import pandas as pd
import pytest

from walmart_sales.data import WalmartDataset


@pytest.fixture
def df_processed() -> pd.DataFrame:
    """Processed weekly sales data."""
    return pd.read_csv(
        "data/test/test_data_processed.csv", parse_dates=["Date"]
    )


@pytest.fixture
def walmart_dataset(df_processed: pd.DataFrame) -> WalmartDataset:
    """Walmart dataset."""
    return WalmartDataset(df_processed, features_window=6, horizon=2)


def test_forecast_week(
    df_processed: pd.DataFrame, walmart_dataset: WalmartDataset
) -> None:
    """lag_1 feature has sales of the last known week before forecast."""
    assert (
        walmart_dataset.df_full.query(
            "Store == 1 and Dept == 1 and forecast_week == '2010-04-23'"
        )["lag_1"]
        == df_processed.query(
            "Store == 1 and Dept == 1 and Date == '2010-04-23'"
        ).Weekly_Sales.iloc[0]
    ).all()


def test_past_weeks(
    df_processed: pd.DataFrame, walmart_dataset: WalmartDataset
) -> None:
    """lag_i feature has sales of the i-th week before forecast."""
    assert (
        walmart_dataset.df_full.query(
            "Store == 1 and Dept == 1 and forecast_week == '2010-04-23'"
        )["lag_4"]
        == df_processed.query(
            "Store == 1 and Dept == 1 and Date == '2010-04-02'"
        ).Weekly_Sales.iloc[0]
    ).all()


def test_targeted_week(walmart_dataset: WalmartDataset) -> None:
    """targeted_week is forecast_week + horizon."""
    assert (
        walmart_dataset.df_full.targeted_week
        == walmart_dataset.df_full.forecast_week
        + pd.to_timedelta(walmart_dataset.df_full.horizon, unit="W")
    ).all()
