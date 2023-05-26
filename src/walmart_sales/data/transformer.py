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
        if is_train:
            y = df["target"]
        else:
            y = None
        return X, y
