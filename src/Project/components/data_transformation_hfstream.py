import os
import sys
import cv2
import json
import random
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from src.Project.logger import logging
from src.Project.exception import CustomException

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = (224, 224)
MAX_SPEED = 120.0
SAFE_DISTANCE_PX = 120
BATCH_SIZE = 512
ENABLE_AUGMENT = True
DEMO_VISUAL = True

INPUT_CSV = "data/hf_data/metadata.csv"
OUTPUT_DIR = "data/processed"
PARQUET_PATH = os.path.join(OUTPUT_DIR, "train_data.parquet")
IMAGE_OUT_DIR = os.path.join(OUTPUT_DIR, "images")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUT_DIR, exist_ok=True)

REQUIRED = [
    "image_front",
    "seg_front",
    "boxes",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "speed_kmh",
    "throttle",
    "steer",
    "brake",
]

# ----------------------------
# UTILITIES
# ----------------------------
def normalize(val, max_val):
    return float(val) / max_val if max_val > 0 else 0.0


def resize_and_normalize(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    orig_h, orig_w = img.shape[:2]
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return img, orig_w, orig_h


def encode_segmentation(seg_path):
    seg = cv2.imread(seg_path)
    if seg is None:
        return None

    seg = cv2.resize(seg, IMG_SIZE)
    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    mask = (gray > 30).astype(np.uint8)
    return mask


def scale_boxes(boxes, orig_w, orig_h):
    if not boxes:
        return []

    sx = IMG_SIZE[0] / orig_w
    sy = IMG_SIZE[1] / orig_h
    scaled = []

    for xmin, ymin, xmax, ymax in boxes:
        scaled.append([xmin * sx, ymin * sy, xmax * sx, ymax * sy])

    return scaled


def compute_nearest_box_distance(boxes):
    if not boxes:
        return 9999.0

    dists = []
    cx_ego = IMG_SIZE[0] / 2
    cy_ego = IMG_SIZE[1]

    for xmin, ymin, xmax, ymax in boxes:
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        dist = np.sqrt((cx_ego - cx) ** 2 + (cy_ego - cy) ** 2)
        dists.append(dist)

    return float(min(dists))


def create_brake_binary(nearest_dist, speed_kmh):
    return 1 if (nearest_dist < SAFE_DISTANCE_PX and speed_kmh > 10) else 0


def random_augment(img):
    if random.random() < 0.3:
        factor = 0.7 + random.random() * 0.6
        img = np.clip(img * factor, 0, 1)

    if random.random() < 0.3:
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = np.clip((img - mean) * 1.2 + mean, 0, 1)

    if random.random() < 0.2:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    return img


def horizontal_flip(img):
    return cv2.flip(img, 1)


def flip_boxes(boxes):
    flipped = []
    for xmin, ymin, xmax, ymax in boxes:
        new_xmin = IMG_SIZE[0] - xmax
        new_xmax = IMG_SIZE[0] - xmin
        flipped.append([new_xmin, ymin, new_xmax, ymax])
    return flipped


def visualize_sample(img, seg, record):
    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    plt.title(
        f"steer={record['steer']:.3f}, "
        f"speed={record['speed_norm']:.2f}, "
        f"brake_bin={record['brake_binary']}"
    )
    plt.axis("off")
    plt.show()


# ----------------------------
# CORE TRANSFORMATION
# ----------------------------
def transform_dataset(max_rows=15000):
    try:
        logging.info("Starting full transformation pipeline")

        df = pd.read_csv(INPUT_CSV)
        df["boxes"] = df["boxes"].apply(eval)

        buffer = []
        saved = 0
        demo_done = False

        for idx, row in df.iterrows():

            if any(pd.isna(row[k]) for k in REQUIRED):
                continue

            # ---- Load + process images ----
            front, orig_w, orig_h = resize_and_normalize(row["image_front"])
            seg = encode_segmentation(row["seg_front"])

            if front is None or seg is None:
                continue

            boxes_scaled = scale_boxes(row["boxes"], orig_w, orig_h)

            if ENABLE_AUGMENT:
                front = random_augment(front)

            # ---- Object proximity ----
            nearest_dist = compute_nearest_box_distance(boxes_scaled)

            # ---- Save transformed images ----
            front_out = os.path.join(IMAGE_OUT_DIR, f"{saved}_front.png")
            seg_out = os.path.join(IMAGE_OUT_DIR, f"{saved}_seg.png")

            cv2.imwrite(front_out, (front * 255).astype(np.uint8))
            cv2.imwrite(seg_out, seg.astype(np.uint8))

            # ---- Normalized scalars ----
            record = {
                "front_img_path": front_out,
                "seg_path": seg_out,
                "velocity_x": normalize(row["velocity_x"], 30),
                "velocity_y": normalize(row["velocity_y"], 30),
                "velocity_z": normalize(row["velocity_z"], 30),
                "speed_norm": normalize(row["speed_kmh"], MAX_SPEED),
                "throttle": row["throttle"],
                "steer": row["steer"],
                "brake": row["brake"],
                "nearest_object_dist": nearest_dist,
                "brake_binary": create_brake_binary(nearest_dist, row["speed_kmh"]),
            }

            buffer.append(record)
            saved += 1

            # ---- LEFT-RIGHT FLIP AUGMENTATION ----
            if ENABLE_AUGMENT and abs(row["steer"]) > 0.15:
                front_flip = horizontal_flip(front)
                seg_flip = horizontal_flip(seg)
                boxes_flip = flip_boxes(boxes_scaled)

                front_fpath = os.path.join(IMAGE_OUT_DIR, f"{saved}_flip_front.png")
                seg_fpath = os.path.join(IMAGE_OUT_DIR, f"{saved}_flip_seg.png")

                cv2.imwrite(front_fpath, (front_flip * 255).astype(np.uint8))
                cv2.imwrite(seg_fpath, (seg_flip * 255).astype(np.uint8))

                nearest_flip = compute_nearest_box_distance(boxes_flip)

                flipped = record.copy()
                flipped["front_img_path"] = front_fpath
                flipped["seg_path"] = seg_fpath
                flipped["steer"] = -row["steer"]
                flipped["velocity_y"] = -record["velocity_y"]
                flipped["nearest_object_dist"] = nearest_flip
                flipped["brake_binary"] = create_brake_binary(
                    nearest_flip, row["speed_kmh"]
                )

                buffer.append(flipped)
                saved += 1

            # ---- Periodic save ----
            if len(buffer) >= BATCH_SIZE:
                table = pa.Table.from_pylist(buffer)
                pq.write_to_dataset(
                    table,
                    root_path=PARQUET_PATH,
                    existing_data_behavior="overwrite_or_ignore"
                )
                buffer.clear()
                logging.info(f"Processed {saved} samples")

            # ---- Visual demo ----
            if DEMO_VISUAL and not demo_done and saved >= 500:
                visualize_sample(front, seg, record)
                demo_done = True

            if saved >= max_rows:
                break

        # ---- Final flush ----
        if buffer:
            table = pa.Table.from_pylist(buffer)
            pq.write_to_dataset(
                table,
                root_path=PARQUET_PATH,
                existing_data_behavior="overwrite_or_ignore"
            )

        logging.info(f"Transformation complete. Total samples: {saved}")

    except Exception as e:
        logging.error("Fatal transformation error", exc_info=True)
        raise CustomException(e, sys)


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    try:
        transform_dataset()
    except Exception as e:
        raise CustomException(e, sys)
