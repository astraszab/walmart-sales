"""LightGBM model for sales forecasting."""

from typing import Any

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

from .model_base import ModelBase
from ..data import WalmartDataTransformer


class WalmartLGBM(ModelBase):
    """LightGBM model for sales forecasting."""

    def __init__(
        self,
        lgbm_regressor: LGBMRegressor,
        transfromer: WalmartDataTransformer,
        differentiate_target: bool = True,
        clip_foreacst: bool = True,
    ):
        """Initialize a model."""
        super().__init__()
        self._lgbm = lgbm_regressor
        self._transformer = transfromer
        self._differentiate_target = differentiate_target
        self._clip_forecast = clip_foreacst

    def _fit(self, df: pd.DataFrame) -> "WalmartLGBM":
        X, y = self._transformer.fit_transform(df)
        if self._differentiate_target:
            y = y - X["lag_1"]
        self._lgbm = self._lgbm.fit(X, y)
        return self

    def _predict(self, df: pd.DataFrame) -> "pd.Series[Any]":
        X = self._transformer.transform(df)
        preds = self._lgbm.predict(X)
        if self._differentiate_target:
            preds = preds + X["lag_1"]
        if self._clip_forecast:
            preds = np.clip(preds, 0, None)
        return pd.Series(preds, index=df.index)
