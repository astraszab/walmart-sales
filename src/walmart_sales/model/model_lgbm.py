"""LightGBM model for sales forecasting."""

from typing import Any

from lightgbm import LGBMRegressor
import pandas as pd

from .model_base import ModelBase
from ..data import WalmartDataTransformer


class WalmartLGBM(ModelBase):
    """LightGBM model for sales forecasting."""

    def __init__(
        self,
        lgbm_regressor: LGBMRegressor,
        transfromer: WalmartDataTransformer,
    ):
        """Initialize a model."""
        super().__init__()
        self._lgbm = lgbm_regressor
        self._transformer = transfromer

    def _fit(self, df: pd.DataFrame) -> "WalmartLGBM":
        X, y = self._transformer.fit_transform(df)
        self._lgbm = self._lgbm.fit(X, y)
        return self

    def _predict(self, df: pd.DataFrame) -> "pd.Series[Any]":
        X = self._transformer.transform(df)
        preds = self._lgbm.predict(X)
        return pd.Series(preds, index=df.index)
