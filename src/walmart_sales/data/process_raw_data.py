"""Process raw data."""

from pathlib import Path
import zipfile

import click
import pandas as pd

from walmart_sales.constants import (
    FEATURES_DATA_FILE,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
    RAW_DATA_ARCHIVE,
    RAW_DATA_DIR,
    STORES_DATA_FILE,
    TRAIN_DATA_FILE,
)


@click.command()
def process_raw_data() -> None:
    """Read raw data, merge, and save the result."""
    with zipfile.ZipFile(Path(RAW_DATA_DIR, RAW_DATA_ARCHIVE), "r") as zip_ref:
        zip_ref.extractall(INTERIM_DATA_DIR)
    df_features = pd.read_csv(Path(INTERIM_DATA_DIR, FEATURES_DATA_FILE))
    df_stores = pd.read_csv(Path(INTERIM_DATA_DIR, STORES_DATA_FILE))
    df_train = pd.read_csv(Path(INTERIM_DATA_DIR, TRAIN_DATA_FILE))
    click.echo("Merging tables...")
    df_merged = df_train.merge(df_stores).merge(df_features)
    processed_dataset_path = Path(PROCESSED_DATA_DIR, PROCESSED_DATA_FILE)
    df_merged.to_csv(processed_dataset_path, index=False)
    click.secho(f"Extracted raw archive to {INTERIM_DATA_DIR}", fg="green")
    click.secho(
        f"Saved merged data to {processed_dataset_path}",
        fg="green",
    )
