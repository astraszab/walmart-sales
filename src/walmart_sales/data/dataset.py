"""Walmart weekly sales dataset."""


from typing import List, Tuple

import pandas as pd

from walmart_sales.constants import (
    FEATURES_KNOWN_IN_ADVANCE,
    FEATURES_STATIONARY,
    FEATURES_WINDOW,
    HORIZON,
    TEST_WEEKS,
    VAL_WEEKS,
)
from walmart_sales.utils import add_weeks, get_week_diff


class WalmartDataset:
    """Walmart weekly sales dataset."""

    def __init__(
        self,
        data: pd.DataFrame,
        features_window: int = FEATURES_WINDOW,
        horizon: int = HORIZON,
        val_weeks: int = VAL_WEEKS,
        test_weeks: int = TEST_WEEKS,
    ):
        """Initialize a dataset.

        Args:
            data: processed data where each row corresponds to a specific
                week and has features associated with it.
            features_window: number of weeks prior to forecast that can
                be used to extract features.
            horizon: number of weeks to forecast.
            val_weeks: number of validation 1-week folds.
            test_weeks: number of test 1-week folds.
        """
        self._features_window = features_window
        self._horizon = horizon
        self._val_weeks = val_weeks
        self._test_weeks = test_weeks
        self._df_full = self._prepare_data(data)

    @property
    def full(self) -> pd.DataFrame:
        """Get full dataset.

        Returns:
            full dataset.
        """
        return self._df_full.copy()

    @property
    def test_ts_split(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Get test folds.

        Returns:
            A list of (train_df, test_df) tuples.
        """
        test_folds = []
        max_week = self._df_full.forecast_week.max()
        for i in range(0, self._test_weeks):
            forecast_week = add_weeks(max_week, -i)
            test_folds.append(
                (
                    self._df_full[self._df_full.forecast_week < forecast_week],
                    self._df_full[
                        self._df_full.forecast_week == forecast_week
                    ],
                )
            )
        return test_folds

    @property
    def val_ts_split(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Get validation folds.

        Returns:
            A list of (train_df, val_df) tuples.
        """
        test_folds = []
        max_week = self._df_full.forecast_week.max()
        for i in range(self._test_weeks, self._test_weeks + self._val_weeks):
            forecast_week = add_weeks(max_week, -i)
            test_folds.append(
                (
                    self._df_full[self._df_full.forecast_week < forecast_week],
                    self._df_full[
                        self._df_full.forecast_week == forecast_week
                    ],
                )
            )
        return test_folds

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[
            data.groupby(["Store", "Dept"])["Date"].transform(
                lambda x: (get_week_diff(x.min(), x.max()) + 1) == x.count()
            )
        ]
        data_pivoted = self._extract_ts_feature(
            data, prefix="lag", target=True
        )
        data_pivoted["targeted_week"] = data_pivoted[
            "forecast_week"
        ] + pd.to_timedelta(data_pivoted.horizon, unit="W")
        data_pivoted = data_pivoted.merge(
            data[FEATURES_KNOWN_IN_ADVANCE + ["Store", "Dept", "Date"]],
            how="left",
            left_on=["Store", "Dept", "targeted_week"],
            right_on=["Store", "Dept", "Date"],
        ).drop("Date", axis=1)
        data_pivoted = data_pivoted.merge(
            data[FEATURES_STATIONARY + ["Store", "Dept", "Date"]],
            how="left",
            left_on=["Store", "Dept", "forecast_week"],
            right_on=["Store", "Dept", "Date"],
        ).drop("Date", axis=1)
        return data_pivoted

    def _extract_ts_feature(
        self, data: pd.DataFrame, prefix: str, target: bool
    ) -> pd.DataFrame:
        df_pivot = data.pivot(
            index=["Store", "Dept"], columns="Date", values=["Weekly_Sales"]
        )
        partial_dfs = []
        for shift in range(0, df_pivot.shape[1] - self._features_window):
            df_features = df_pivot.iloc[
                :, range(shift, shift + self._features_window)
            ]
            forecast_week = df_features.columns[-1][1]
            df_features.columns = [
                f"{prefix}_{i}" for i in range(self._features_window, 0, -1)
            ]
            df_features = df_features.reset_index()
            if target:
                df_target = df_pivot.iloc[
                    :,
                    range(
                        shift + self._features_window,
                        min(
                            shift + self._features_window + self._horizon,
                            df_pivot.shape[1],
                        ),
                    ),
                ]
                df_target.columns = range(1, len(df_target.columns) + 1)
                df_target = df_target.stack().reset_index()
                df_target.columns = list(df_target.columns[:-2]) + [
                    "horizon",
                    "target",
                ]
                df_features = df_features.merge(df_target)
            df_features["forecast_week"] = forecast_week
            partial_dfs.append(df_features)
        return pd.concat(partial_dfs, ignore_index=True)
