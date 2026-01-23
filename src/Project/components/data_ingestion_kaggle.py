# Importing necessary libraries
import os
import sys
import pandas as pd
import ntpath
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# Importing custom components
from src.Project.logger import logging
from src.Project.exception import CustomException

@dataclass
class DataIngestionConfig:
    """Configuration for data paths."""
    
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def path_leaf(self, path):
        """
        Extracts only the filename from a full path. 
        Necessary because the dataset CSV often contains local paths from the recording PC.
        """
        head, tail = ntpath.split(path)
        return tail

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion component")
        try:
            # Define the source directory
            data_dir = r"C:\Users\Nandita\Downloads\End-to-End-Deep-Learning-for-Autonomous-Driving\data\track"
            columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
            
            # Read the driving log
            logging.info(f"Reading dataset from {data_dir}")
            df = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'), names=columns)
            
            # Clean the image paths 
            logging.info("Cleaning image paths in the dataframe")
            df['center'] = df['center'].apply(self.path_leaf)
            df['left'] = df['left'].apply(self.path_leaf)
            df['right'] = df['right'].apply(self.path_leaf)

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the cleaned raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Train-Test Split
            logging.info("Initiating Train-Test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()