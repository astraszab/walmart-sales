"""Plot forecast vs target."""


import pandas as pd


from walmart_sales.constants import FEATURES_WINDOW
from walmart_sales.utils import add_weeks


def plot_forecast_vs_target(
    df: pd.DataFrame,
    store: int,
    dept: int,
    forecast_week_str: str,
    features_window: int = FEATURES_WINDOW,
) -> None:
    """Plot forecast vs target.

    Args:
        df: dataframe with lags, target, and forecast.
        store: store number.
        dept: department number.
        forecast_week_str: week at which forecast was
            obtained, e.g., '2012-09-07'.
        features_window: number of lags in the data.

    Raises:
        ValueError: if there is no such department forecasted
            at a given week in data.
    """
    forecast_week = pd.Timestamp(forecast_week_str)
    sub_df = df[
        (df.Store == store)
        & (df.Dept == dept)
        & (df.forecast_week == forecast_week)
    ]
    if len(sub_df) == 0:
        raise ValueError("No such department in a dataset.")
    max_horizon = sub_df.horizon.max()
    vis_df = pd.DataFrame(
        list(
            sub_df.iloc[0][(f"lag_{i}" for i in range(features_window, 0, -1))]
        )
        + list(sub_df["target"]),
        columns=["target"],
    ).fillna(0)
    vis_df["forecast"] = sub_df.pred.set_axis(
        [i for i in range(features_window, features_window + max_horizon)]
    )
    vis_df["forecast"] = vis_df["forecast"].fillna(vis_df["target"])
    vis_df = vis_df.set_axis(
        pd.date_range(
            add_weeks(forecast_week, -features_window + 1),
            periods=len(vis_df),
            freq="w",
        )
    )
    title = (
        f"Store {store}\nDepartment {dept}\n"
        f"forecast obtained in {forecast_week.strftime('%Y-%m-%d')}"
    )
    vis_df[["forecast", "target"]].plot(
        title=title, xlabel="date", ylabel="weekly sales"
    )
