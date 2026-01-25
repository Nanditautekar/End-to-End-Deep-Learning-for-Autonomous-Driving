import os
import sys
import json
import pandas as pd
from datasets import load_dataset
from PIL import Image
import numpy as np

from src.Project.logger import logging
from src.Project.exception import CustomException

# CONFIG

DATASET_NAME = "immanuelpeter/carla-autopilot-multimodal-dataset"
SPLIT = "train"

OUTPUT_DIR = "data/hf_data"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

CSV_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")
JSON_PATH = os.path.join(OUTPUT_DIR, "dataset.json")
CKPT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.txt")

MAX_SAMPLES = 10000
FLUSH_EVERY = 200

os.makedirs(IMAGE_DIR, exist_ok=True)

# Utils

def save_image(img_obj, filename):
    if img_obj is None:
        return None
    try:
        if isinstance(img_obj, Image.Image):
            img = img_obj
        else:
            img = Image.fromarray(np.array(img_obj))

        if img.mode != "RGB":
            img = img.convert("RGB")

        path = os.path.join(IMAGE_DIR, filename)
        img.save(path)
        return path

    except Exception as e:
        logging.error(f"Image save failed: {filename} → {e}")
        return None


def get_checkpoint():
    if not os.path.exists(CKPT_PATH):
        return 0
    with open(CKPT_PATH, "r") as f:
        return int(f.read().strip())


def save_checkpoint(idx):
    with open(CKPT_PATH, "w") as f:
        f.write(str(idx))


def load_existing_json():
    if not os.path.exists(JSON_PATH):
        return []
    with open(JSON_PATH, "r") as f:
        return json.load(f)


def append_json(records):
    existing = load_existing_json()
    existing.extend(records)
    with open(JSON_PATH, "w") as f:
        json.dump(existing, f, indent=2)


def append_csv(rows):
    write_header = not os.path.exists(CSV_PATH)
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, mode="a", header=write_header, index=False)

# Main

def main():
    try:
        logging.info("Loading dataset in streaming mode...")
        dataset = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)

        start_raw = get_checkpoint()
        logging.info(f"Resuming from raw index {start_raw}")

        json_buffer = []
        csv_buffer = []

        saved = len(load_existing_json())

        for raw_idx, sample in enumerate(dataset):

            if raw_idx < start_raw:
                continue

            # Minimal safe requirements 
            if sample.get("image_front_right") is None:
                continue

            # Save images 
            front_right_path = save_image(
                sample.get("image_front_right"),
                f"{saved}_front_right.png"
            )

            seg_front_path = save_image(
                sample.get("seg_front"),
                f"{saved}_seg_front.png"
            )

            front_path = save_image(
                sample.get("image_front"),
                f"{saved}_front.png"
            )

            left_path = save_image(
                sample.get("image_front_left"),
                f"{saved}_left.png"
            )

            if not front_right_path:
                continue

            # JSON record 
            record = {
                "image_front_right": front_right_path,
                "image_front": front_path,
                "image_front_left": left_path,
                "seg_front": seg_front_path,
                "boxes": sample.get("boxes"),
                "box_labels": sample.get("box_labels"),
                "velocity_x": sample.get("velocity_x"),
                "velocity_y": sample.get("velocity_y"),
                "velocity_z": sample.get("velocity_z"),
                "speed_kmh": sample.get("speed_kmh"),
                "throttle": sample.get("throttle"),
                "steer": sample.get("steer"),
                "brake": sample.get("brake"),
            }

            json_buffer.append(record)

            # CSV
            csv_buffer.append({
                "image_front_right": front_right_path,
                "image_front": front_path,
                "image_front_left": left_path,
                "seg_front": seg_front_path,
                "velocity_x": record["velocity_x"],
                "velocity_y": record["velocity_y"],
                "velocity_z": record["velocity_z"],
                "speed_kmh": record["speed_kmh"],
                "throttle": record["throttle"],
                "steer": record["steer"],
                "brake": record["brake"],
            })

            saved += 1
            save_checkpoint(raw_idx + 1)

            # Progress
            if saved % 100 == 0:
                print(f"{saved} samples completed")
                logging.info(f"{saved} samples completed")

            # Flush 
            if len(json_buffer) >= FLUSH_EVERY:
                append_json(json_buffer)
                append_csv(csv_buffer)
                json_buffer.clear()
                csv_buffer.clear()
                logging.info(f"Flushed {FLUSH_EVERY} samples to disk")

            if saved >= MAX_SAMPLES:
                break

        # Final flush
        if json_buffer:
            append_json(json_buffer)
            append_csv(csv_buffer)

        logging.info(f"Done. Total samples: {saved}")
        print("=====================================")
        print("Download complete!")
        print(f"Images saved to: {IMAGE_DIR}")
        print(f"Single JSON:     {JSON_PATH}")
        print(f"Metadata CSV:   {CSV_PATH}")
        print(f"Total rows:     {saved}")
        print("=====================================")

    except Exception as e:
        logging.error("Fatal ingestion error", exc_info=True)
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
