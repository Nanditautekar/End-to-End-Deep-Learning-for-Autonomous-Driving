import os
import sys
import csv
import pandas as pd
from datasets import load_dataset
from PIL import Image

from src.Project.logger import logging
from src.Project.exception import CustomException


DATASET_NAME = "immanuelpeter/carla-autopilot-images"

SAVE_DIR = "data/hf_data/train"
IMG_DIR = os.path.join(SAVE_DIR, "images")
CSV_PATH = os.path.join(SAVE_DIR, "labels.csv")
CKPT_PATH = os.path.join(SAVE_DIR, "checkpoint.txt")

COLUMNS = [
    "image_front",
    "image_front_left",
    "image_front_right",
    "speed_kmh",
    "throttle",
    "steer",
    "brake",
]

MAX_SAMPLES = 10000


def save_image(img, path):
    try:
        if not isinstance(img, Image.Image):
            logging.warning(f"Invalid image object at {path}")
            return False

        if img.mode == "RGBA":
            img = img.convert("RGB")

        img.save(path, format="JPEG", quality=95)
        return True

    except Exception as e:
        logging.error(f"Failed to save image {path}: {e}")
        return False


def get_existing_count():
    try:
        if not os.path.exists(CSV_PATH):
            return 0
        df = pd.read_csv(CSV_PATH)
        return len(df)

    except Exception as e:
        raise CustomException(e, sys)


def get_checkpoint():
    try:
        if not os.path.exists(CKPT_PATH):
            return 0
        with open(CKPT_PATH, "r") as f:
            return int(f.read().strip())

    except Exception as e:
        raise CustomException(e, sys)


def save_checkpoint(raw_index):
    try:
        with open(CKPT_PATH, "w") as f:
            f.write(str(raw_index))

    except Exception as e:
        raise CustomException(e, sys)


def main():
    try:
        os.makedirs(IMG_DIR, exist_ok=True)

        logging.info("Starting HuggingFace dataset ingestion")

        dataset = load_dataset(
            DATASET_NAME,
            split="train",
            streaming=True
        )

        start_saved = get_existing_count()
        start_raw = get_checkpoint()

        logging.info(f"Resuming download")
        logging.info(f"Saved samples so far: {start_saved}")
        logging.info(f"Raw stream index: {start_raw}")

        rows = []
        saved = start_saved

        for raw_idx, sample in enumerate(dataset):

            # Skip raw rows until checkpoint
            if raw_idx < start_raw:
                continue

            # Skip invalid rows
            if any(sample[k] is None for k in COLUMNS):
                continue

            front_path = os.path.join(IMG_DIR, f"{saved}_front.jpg")
            left_path  = os.path.join(IMG_DIR, f"{saved}_left.jpg")
            right_path = os.path.join(IMG_DIR, f"{saved}_right.jpg")

            ok1 = save_image(sample["image_front"], front_path)
            ok2 = save_image(sample["image_front_left"], left_path)
            ok3 = save_image(sample["image_front_right"], right_path)

            if not (ok1 and ok2 and ok3):
                logging.warning(f"Skipping sample {saved} due to image save failure")
                continue

            rows.append({
                "front_img": front_path,
                "left_img": left_path,
                "right_img": right_path,
                "speed_kmh": sample["speed_kmh"],
                "throttle": sample["throttle"],
                "steer": sample["steer"],
                "brake": sample["brake"],
            })

            saved += 1
            save_checkpoint(raw_idx + 1)

            if saved % 100 == 0:
                logging.info(f"Downloaded {saved} samples")

            if saved >= MAX_SAMPLES:
                logging.info("Reached MAX_SAMPLES limit")
                break

        if rows:
            write_header = not os.path.exists(CSV_PATH)

            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)

            logging.info(f"Appended {len(rows)} rows to CSV")

        logging.info(f"Download completed. Total samples: {saved}")
        logging.info(f"Images directory: {IMG_DIR}")
        logging.info(f"CSV path: {CSV_PATH}")
        logging.info(f"Checkpoint path: {CKPT_PATH}")

        print(f"\n✅ Now have {saved} total samples")
        print(f"Images → {IMG_DIR}")
        print(f"CSV → {CSV_PATH}")
        print(f"Checkpoint → {CKPT_PATH}")

    except Exception as e:
        logging.error("Fatal error in HF data ingestion", exc_info=True)
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
