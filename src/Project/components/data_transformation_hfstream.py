import os
import sys
import json
import random

from src.Project.logger import logging
from src.Project.exception import CustomException

# PROJECT ROOT (VERY IMPORTANT)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

# PATHS

INPUT_JSON = os.path.join(PROJECT_ROOT, "data", "hf_data", "dataset.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed_json")

TRAIN_JSON = os.path.join(OUTPUT_DIR, "train.json")
VAL_JSON   = os.path.join(OUTPUT_DIR, "val.json")
TEST_JSON  = os.path.join(OUTPUT_DIR, "test.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# SPLIT CONFIG

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

random.seed(42)

# REQUIRED IMAGE KEYS

IMAGE_KEYS = [
    "image_front",
    "image_front_left",
    "image_front_right",
    "seg_front"
]

# UTILS

def resolve_path(p):
    """
    Convert relative dataset paths into absolute OS-safe paths
    """
    if p is None:
        return None
    p = p.replace("\\", os.sep).replace("/", os.sep)
    return os.path.join(PROJECT_ROOT, p)


def paths_exist(row):
    """
    Check that all image paths actually exist on disk
    """
    for k in IMAGE_KEYS:
        if k not in row:
            return False
        if row[k] is None:
            return False
        abs_path = resolve_path(row[k])
        if not os.path.exists(abs_path):
            return False
    return True


def nearest_object_distance(boxes, img_w=224, img_h=224):
    """
    Distance from ego vehicle (bottom-center) to nearest box center.
    Used only as metadata — no image ops.
    """
    if not boxes:
        return 9999.0

    ego_x = img_w / 2
    ego_y = img_h

    min_dist = 9999.0
    for xmin, ymin, xmax, ymax in boxes:
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        dist = ((ego_x - cx) ** 2 + (ego_y - cy) ** 2) ** 0.5
        min_dist = min(min_dist, dist)

    return float(min_dist)

# CORE TRANSFORMATION

def transform_to_json():
    try:
        logging.info("Starting JSON transformation pipeline")

        with open(INPUT_JSON, "r") as f:
            data = json.load(f)

        logging.info(f"Total raw rows in dataset.json: {len(data)}")

        valid_rows = []
        skipped_missing_paths = 0
        skipped_invalid_boxes = 0

        for row in data:

            # ---- Path validation ----
            if not paths_exist(row):
                skipped_missing_paths += 1
                continue

            # ---- Boxes validation ----
            boxes = row.get("boxes", [])
            if not isinstance(boxes, list):
                skipped_invalid_boxes += 1
                continue

            # ---- Derived metadata (NO image loading) ----
            row["nearest_object_dist"] = nearest_object_distance(boxes)
            row["box_count"] = len(boxes)

            valid_rows.append(row)

        logging.info(f"Valid samples kept: {len(valid_rows)}")
        logging.info(f"Skipped (missing images): {skipped_missing_paths}")
        logging.info(f"Skipped (invalid boxes): {skipped_invalid_boxes}")

        # TRAIN / VAL / TEST SPLIT
    
        random.shuffle(valid_rows)

        n = len(valid_rows)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        train_data = valid_rows[:n_train]
        val_data   = valid_rows[n_train:n_train + n_val]
        test_data  = valid_rows[n_train + n_val:]

        # SAVE JSON FILES
        
        with open(TRAIN_JSON, "w") as f:
            json.dump(train_data, f, indent=2)

        with open(VAL_JSON, "w") as f:
            json.dump(val_data, f, indent=2)

        with open(TEST_JSON, "w") as f:
            json.dump(test_data, f, indent=2)

        logging.info("JSON transformation complete")
        logging.info(f"Train samples: {len(train_data)}")
        logging.info(f"Val samples:   {len(val_data)}")
        logging.info(f"Test samples:  {len(test_data)}")

    except Exception as e:
        logging.error("Fatal error in transformation", exc_info=True)
        raise CustomException(e, sys)


# ENTRY POINT

if __name__ == "__main__":
    transform_to_json()
