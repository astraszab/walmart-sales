"""Model validation."""

from collections.abc import Iterable
from typing import Any, Callable, Tuple

import pandas as pd

from walmart_sales.model import ModelBase


Metric = Callable[["pd.Series[Any]", "pd.Series[Any]"], float]


def validate(
    val_folds: Iterable[Tuple[pd.DataFrame, pd.DataFrame]],
    model: ModelBase,
    metric: Metric,
    verbose: bool = False,
) -> Tuple[float, pd.DataFrame]:
    """Validate a model.

    Args:
        val_folds: list of (train_df, val_df) tuples.
        model: model to validate.
        metric: metric to calculate for validation set.
        verbose: if True, fold names will be printed.

    Returns:
        Metric calculated for validation set and dataframe with predictions.
    """
    pred_dfs = []
    for i, (train_fold, val_fold) in enumerate(val_folds, 1):
        if verbose:
            print(f"Validating fold {i}...")
        model = model.fit(train_fold)
        pred_df = val_fold.copy()
        pred_df["pred"] = model.predict(val_fold.drop("target", axis=1))
        pred_dfs.append(pred_df)
    pred_df = pd.concat(pred_dfs)
    return metric(pred_df["target"], pred_df["pred"]), pred_df
