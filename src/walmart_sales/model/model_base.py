"""Model abstract class."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar


import pandas as pd


T = TypeVar("T", bound="ModelBase")


class ModelBase(ABC):
    """Model abstract class."""

    def __init__(self) -> None:
        """Initialize a model."""
        self._is_fitted = False

    def fit(self: T, df: pd.DataFrame) -> T:
        """Fit a model.

        Args:
            df: train set with 'target' column and features.

        Returns:
            Trained model.
        """
        fitted_model = self._fit(df)
        self._is_fitted = True
        return fitted_model

    def predict(self, df: pd.DataFrame) -> "pd.Series[Any]":
        """Predict with a model."""
        if not self._is_fitted:
            raise ValueError(
                f"{self.__class__.__name__} must be fit before calling predict"
            )
        return self._predict(df)

    @property
    def is_fitted(self) -> bool:
        """Return True if the model is fitted."""
        return self._is_fitted

    @abstractmethod
    def _fit(self: T, df: pd.DataFrame) -> T:
        pass

    @abstractmethod
    def _predict(self, df: pd.DataFrame) -> "pd.Series[Any]":
        pass
