"""Baseline no-ML models."""

from typing import Any

import pandas as pd

from .model_base import ModelBase


class LastWeekModel(ModelBase):
    """Forecast is sales for the last week available."""

    def _fit(self, df: pd.DataFrame) -> "LastWeekModel":
        return self

    def _predict(self, df: pd.DataFrame) -> "pd.Series[Any]":
        return df["lag_1"].fillna(0)
