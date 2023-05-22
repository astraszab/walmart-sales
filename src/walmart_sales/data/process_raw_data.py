"""Process raw data."""

import zipfile


import click
import pandas as pd


@click.command()
def process_raw_data() -> None:
    """Read raw data, merge, and save the result."""
    with zipfile.ZipFile(
        "data/raw/walmart-recruiting-sales-in-stormy-weather.zip", "r"
    ) as zip_ref:
        zip_ref.extractall("data/interim")
    df_train = pd.read_csv("data/interim/train.csv.zip")
    df_weather = pd.read_csv("data/interim/weather.csv.zip")
    df_key = pd.read_csv("data/interim/key.csv.zip")
    click.echo("Merging tables...")
    df_merged = df_train.merge(df_key).merge(df_weather)
    click.echo("Removing items with zero sales...")
    df_processed = df_merged[
        df_merged.groupby(["item_nbr", "store_nbr"], as_index=False)[
            "units"
        ].transform("sum")
        != 0
    ]
    df_processed.to_csv("data/processed/df_full.csv")
    click.secho("Extracted data to data/interim/", fg="green")
    click.secho("Saved merged data to data/processed/df_full.csv", fg="green")
