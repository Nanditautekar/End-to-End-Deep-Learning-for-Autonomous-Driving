# Import necessary libraries
import os
import sys
import pandas as pd
import numpy as np
import ntpath
from sklearn.utils import shuffle

# Import custom logging and exception handling modules
from src.Project.logger import logging
from src.Project.exception import CustomException


def load_driving_log():
    """
    Loads the driving_log.csv file into a pandas DataFrame.
    """
    try:
        logging.info("Starting to load driving_log.csv")

        # Dataset base path 
        df_path = r"C:\Users\Nandita\Downloads\End-to-End-Deep-Learning-for-Autonomous-Driving\data\track"
        csv_path = os.path.join(df_path, "driving_log.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"driving_log.csv not found at: {csv_path}")

        # Required column Names 
        columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
        data = pd.read_csv(csv_path, names=columns)

        data["steering"] = pd.to_numeric(data["steering"], errors="coerce")

        pd.set_option('display.max_colwidth', None)

        logging.info(f"driving_log.csv loaded successfully with shape: {data.shape}")
        logging.info(f"Sample rows:\n{data.head()}")

        return data

    except Exception as e:
        logging.error("Error occurred while loading driving_log.csv")
        raise CustomException(e, sys)


def clean_image_paths(data):
    """
    Extracts only filenames from full image paths.
    Logic kept exactly the same.
    """
    try:
        logging.info("Cleaning image file paths")

        def path_leaf(path):
            head, tail = ntpath.split(path)
            return tail

        data['center'] = data['center'].apply(path_leaf)
        data['left']   = data['left'].apply(path_leaf)
        data['right']  = data['right'].apply(path_leaf)

        logging.info("Image paths cleaned successfully")
        logging.info(f"Sample rows after path cleaning:\n{data.head()}")

        return data

    except Exception as e:
        logging.error("Error occurred while cleaning image paths")
        raise CustomException(e, sys)


def balance_steering_data(data, num_bins=25, samples_per_bin=400):
    """
    Balances steering distribution by limiting samples per bin.
    Logic is kept exactly the same.
    """
    try:
        logging.info("Starting steering angle balancing process")

        hist, bins = np.histogram(data['steering'], num_bins)

        remove_list = []

        for j in range(num_bins):
            list_ = []

            for i in range(len(data['steering'])):
                if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j + 1]:
                    list_.append(i)

            list_ = shuffle(list_)
            list_ = list_[samples_per_bin:]
            remove_list.extend(list_)

        logging.info(f"Total samples marked for removal: {len(remove_list)}")
        print('removed:', len(remove_list))

        data.drop(data.index[remove_list], inplace=True)

        logging.info(f"Remaining samples after balancing: {len(data)}")
        print('remaining:', len(data))

        return data

    except Exception as e:
        logging.error("Error occurred during steering angle balancing")
        raise CustomException(e, sys)
    
def plot_steering_distribution(data, num_bins=25, samples_per_bin=400):
    """
    Plots steering angle distribution after balancing.
    """
    try:
        logging.info("Plotting steering distribution after balancing")

        hist, bins = np.histogram(data["steering"], num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5

        plt.figure(figsize=(10, 5))
        plt.bar(center, hist, width=(bins[1] - bins[0]))
        plt.plot(
            (np.min(data["steering"]), np.max(data["steering"])),
            (samples_per_bin, samples_per_bin),
            linestyle="--"
        )
        plt.title("Steering Distribution After Balancing")
        plt.xlabel("Steering Angle")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

        print("Post-balancing bin counts:", hist.tolist())
        print("Max bin count:", hist.max())
        print("Remaining samples:", len(data))
        print("Zero steering count:", (data["steering"] == 0).sum())

        logging.info("Post-balancing histogram plotted successfully")
        logging.info(f"Max bin count after balancing: {hist.max()}")
        logging.info(f"Remaining samples after balancing: {len(data)}")
        logging.info(f"Zero steering count after balancing: {(data['steering'] == 0).sum()}")

        return data

    except Exception as e:
        logging.error("Error while plotting post-balancing steering histogram")
        raise CustomException(e, sys)



if __name__ == "__main__":
    # Load → Clean Paths → Balance Steering

    data = load_driving_log()
    data = clean_image_paths(data)
    data = balance_steering_data(data)

    print(data.head())


     