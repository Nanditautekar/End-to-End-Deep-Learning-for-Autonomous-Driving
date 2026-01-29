import os
import sys
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from src.Project.logger import logging
from src.Project.exception import CustomException

MASTER_PARQUET = "data/processed/master.parquet"
SPLIT_DIR = "data/processed/splits"

TRAIN_PARQUET = os.path.join(SPLIT_DIR, "train.parquet")
VAL_PARQUET   = os.path.join(SPLIT_DIR, "val.parquet")
TEST_PARQUET  = os.path.join(SPLIT_DIR, "test.parquet")

os.makedirs(SPLIT_DIR, exist_ok=True)


def make_strat_columns(df):
    """Create multi-factor stratification columns"""

    # Steering bucket
    def steer_bucket(x):
        if x < -0.15:
            return "left"
        elif x > 0.15:
            return "right"
        return "straight"

    # Speed bucket
    def speed_bucket(x):
        if x < 0.3:
            return "slow"
        elif x < 0.6:
            return "medium"
        return "fast"

    df["steer_bin"] = df["steer"].apply(steer_bucket)
    df["speed_bin"] = df["speed_norm"].apply(speed_bucket)

    # Combined strat key
    df["strat_key"] = (
        df["brake_binary"].astype(str) + "_" +
        df["steer_bin"] + "_" +
        df["speed_bin"]
    )

    return df


def log_distribution(df, name):
    logging.info(f"{name} distribution:")
    logging.info(df["brake_binary"].value_counts(normalize=True))
    logging.info(df["steer_bin"].value_counts(normalize=True))
    logging.info(df["speed_bin"].value_counts(normalize=True))


def split_dataset(test_size=0.15, val_size=0.15):
    try:
        logging.info("Loading master dataset")

        df = pq.read_table(MASTER_PARQUET).to_pandas()

        logging.info("Creating stratification columns")
        df = make_strat_columns(df)

        logging.info("Performing stratified split")

        train_df, temp_df = train_test_split(
            df,
            test_size=(test_size + val_size),
            stratify=df["strat_key"],
            random_state=42,
            shuffle=True
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size / (test_size + val_size),
            stratify=temp_df["strat_key"],
            random_state=42,
            shuffle=True
        )

        # Drop helper columns
        for d in (train_df, val_df, test_df):
            d.drop(columns=["steer_bin", "speed_bin", "strat_key"], inplace=True)

        # Save splits
        train_df.to_parquet(TRAIN_PARQUET)
        val_df.to_parquet(VAL_PARQUET)
        test_df.to_parquet(TEST_PARQUET)

        # Logging
        logging.info("Split complete")
        logging.info(f"Train: {len(train_df)}")
        logging.info(f"Val:   {len(val_df)}")
        logging.info(f"Test:  {len(test_df)}")

        # Distribution sanity check
        log_distribution(train_df.assign(
            steer_bin=make_strat_columns(train_df)["steer_bin"],
            speed_bin=make_strat_columns(train_df)["speed_bin"]
        ), "TRAIN")

        log_distribution(val_df.assign(
            steer_bin=make_strat_columns(val_df)["steer_bin"],
            speed_bin=make_strat_columns(val_df)["speed_bin"]
        ), "VAL")

        log_distribution(test_df.assign(
            steer_bin=make_strat_columns(test_df)["steer_bin"],
            speed_bin=make_strat_columns(test_df)["speed_bin"]
        ), "TEST")

    except Exception as e:
        logging.error("Fatal split error", exc_info=True)
        raise CustomException(e, sys)


if __name__ == "__main__":
    split_dataset()
