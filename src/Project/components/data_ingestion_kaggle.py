import os 
import sys
from dataclasses import dataclass
from src.Project.exception import CustomException
from src.Project.logger import logging
import zipfile
from pathlib import Path
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi


@dataclass
class DataIngestionConfig:
    dataset_name: str = "zahidbooni/alltownswithweather"
    raw_data_path: str = os.path.join("data", "raw")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        try:
            logging.info("Starting data ingestion process")

            os.makedirs(self.config.raw_data_path, exist_ok=True)
            logging.info(f"Created directory at {self.config.raw_data_path}")

            api = KaggleApi()
            api.authenticate()
            logging.info("Kaggle API authenticated successfully")

            logging.info(
                f"Downloading dataset: {self.config.dataset_name}"
            )

            api.dataset_download_files(
                self.config.dataset_name,
                path=self.config.raw_data_path,
                unzip=True
            )

            logging.info("Dataset downloaded and extracted successfully")

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    config = DataIngestionConfig()
    data_ingestion = DataIngestion(config)
    data_ingestion.download_data()