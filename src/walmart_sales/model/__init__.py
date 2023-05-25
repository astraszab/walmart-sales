"""Forecasting models."""

from .baseline_models import LastWeekModel
from .model_base import ModelBase


__all__ = ["ModelBase", "LastWeekModel"]
