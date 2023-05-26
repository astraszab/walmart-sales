"""Constants to use in the package."""


from pathlib import Path


# Task specific parameters
FEATURES_WINDOW = 12
HORIZON = 4

# Validation and test
TEST_WEEKS = 12
VAL_WEEKS = 12

# Features
FEATURES_STATIONARY = ["Type", "Size"]

# Data paths
RAW_DATA_DIR = Path("data/raw")
INTERIM_DATA_DIR = Path("data/interim")
PROCESSED_DATA_DIR = Path("data/processed")
RAW_DATA_ARCHIVE = Path("archive.zip")
FEATURES_DATA_FILE = Path("features.csv")
STORES_DATA_FILE = Path("stores.csv")
TRAIN_DATA_FILE = Path("train.csv")
PROCESSED_DATA_FILE = Path("df_full.csv")
