"""Walmart weekly sales dataset."""


import pandas as pd

from walmart_sales.constants import FEATURES_WINDOW, HORIZON
from walmart_sales.utils import get_week_diff


class WalmartDataset:
    """Walmart weekly sales dataset."""

    def __init__(
        self,
        data: pd.DataFrame,
        features_window: int = FEATURES_WINDOW,
        horizon: int = HORIZON,
    ):
        """Initialize a dataset.

        Args:
            data: processed data where each row corresponds to a specific
                week and has features associated with it.
            features_window: number of weeks prior to forecast that can
                be used to extract features.
            horizon: number of weeks to forecast.
        """
        self.features_window = features_window
        self.horizon = horizon
        self.df_full = self._prepare_data(data)

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[
            data.groupby(["Store", "Dept"])["Date"].transform(
                lambda x: (get_week_diff(x.min(), x.max()) + 1) == x.count()
            )
        ]
        data_pivoted = self._extract_ts_feature(
            data, prefix="lag", target=True
        )
        return data_pivoted

    def _extract_ts_feature(
        self, data: pd.DataFrame, prefix: str, target: bool
    ) -> pd.DataFrame:
        df_pivot = data.pivot(
            index=["Store", "Dept"], columns="Date", values=["Weekly_Sales"]
        )
        partial_dfs = []
        for shift in range(0, df_pivot.shape[1] - self.features_window):
            df_features = df_pivot.iloc[
                :, range(shift, shift + self.features_window)
            ]
            forecast_week = df_features.columns[-1][1]
            df_features.columns = [
                f"{prefix}_{i}" for i in range(self.features_window, 0, -1)
            ]
            df_features = df_features.reset_index()
            if target:
                df_target = df_pivot.iloc[
                    :,
                    range(
                        shift + self.features_window,
                        min(
                            shift + self.features_window + self.horizon,
                            df_pivot.shape[1],
                        ),
                    ),
                ]
                df_target.columns = [
                    str(i) for i in range(1, len(df_target.columns) + 1)
                ]
                df_target = df_target.stack().reset_index()
                df_target.columns = list(df_target.columns[:-2]) + [
                    "horizon",
                    "target",
                ]
                df_features = df_features.merge(df_target)
            df_features["forecast_week"] = forecast_week
            partial_dfs.append(df_features)
        return pd.concat(partial_dfs, ignore_index=True)
