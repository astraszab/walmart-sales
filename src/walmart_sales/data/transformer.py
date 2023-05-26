"""Transforms and extracts features and target from data."""

from typing import Any, Optional, Tuple

import pandas as pd

from walmart_sales.constants import FEATURES_WINDOW


class WalmartDataTransformer:
    """Transforms and extracts features and target from data."""

    def __init__(
        self,
        features_window: int = FEATURES_WINDOW,
    ):
        """Initialize transformer."""
        self._features_window = features_window
        self._fit = False

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, "pd.Series[Any]"]:
        """Fit transformer and transform train data."""
        X, y = self._get_X_y(df, is_train=True)
        self._fit = True
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        if not self._fit:
            raise RuntimeError(
                "The transformer is not fit. "
                "Please call fit_transform method first."
            )
        return self._get_X_y(df, is_train=False)[0]

    def _get_X_y(
        self, df: pd.DataFrame, is_train: bool
    ) -> Tuple[pd.DataFrame, Optional["pd.Series[Any]"]]:
        lags = [f"lag_{i}" for i in range(self._features_window, 0, -1)]
        X = df[lags].fillna(0)
        X["horizon"] = df["horizon"]
        X["store_size"] = df["Size"]
        X["store_type_A"] = df["Type"] == "A"
        X["store_type_B"] = df["Type"] == "B"
        X["is_holiday"] = df["IsHoliday"]
        X["targeted_week_number"] = (
            df["targeted_week"].dt.isocalendar().week.astype(int)
        )
        X["forecast_week_number"] = (
            df["forecast_week"].dt.isocalendar().week.astype(int)
        )
        deltas = X[lags].diff(axis=1).iloc[:, 1:]
        X[[f"delta_{i}" for i in range(deltas.shape[1], 0, -1)]] = deltas
        X["temperature_delta"] = (
            df["Temperature_lag_2"] - df["Temperature_lag_1"]
        )
        X["fuel_price_delta"] = df["Fuel_Price_lag_2"] - df["Fuel_Price_lag_1"]
        X["cpi_delta"] = df["CPI_lag_2"] - df["CPI_lag_1"]
        X["unemployment_delta"] = (
            df["Unemployment_lag_2"] - df["Unemployment_lag_1"]
        )
        for i in range(1, 6, 1):
            X[f"markdown{i}_lag_1"] = df[f"MarkDown{i}_lag_1"].fillna(0)
        if is_train:
            y = df["target"]
        else:
            y = None
        return X, y
