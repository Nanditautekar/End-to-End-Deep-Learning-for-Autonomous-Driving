import os
import sys
import json
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset

from src.Project.logger import logging
from src.Project.exception import CustomException


class SelfDrivingDataset(Dataset):

    def __init__(
        self,
        json_path: str,
        project_root: str,
        img_size=(224, 224),
        transform=None
    ):
        try:
            self.project_root = project_root
            self.img_size = img_size
            self.transform = transform

            logging.info(f"Loading dataset from: {json_path}")

            with open(json_path, "r") as f:
                self.data = json.load(f)

            if len(self.data) < 2:
                raise ValueError("Dataset must contain at least 2 samples for t→t+1 prediction")

            logging.info(f"Loaded {len(self.data)} raw samples")

        except Exception as e:
            logging.error("Error initializing SelfDrivingDataset", exc_info=True)
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # IMPORTANT: length is N-1 because we predict next timestep
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data) - 1

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _resolve_path(self, rel_path: str) -> str:
        if rel_path is None:
            raise ValueError("Received None image path")

        rel_path = rel_path.replace("\\", os.sep).replace("/", os.sep)
        abs_path = os.path.join(self.project_root, rel_path)

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Missing image file: {abs_path}")

        return abs_path

    def _load_image(self, rel_path: str, is_seg: bool = False):
        """
        Load and resize image.
        Segmentation masks use nearest interpolation.
        """
        abs_path = self._resolve_path(rel_path)

        if is_seg:
            img = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to read segmentation image: {abs_path}")

            img = cv2.resize(
                img,
                self.img_size,
                interpolation=cv2.INTER_NEAREST
            )

            # Ensure binary or class-index mask
            img = img.astype(np.int64)

        else:
            img = cv2.imread(abs_path)
            if img is None:
                raise ValueError(f"Failed to read RGB image: {abs_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img.astype(np.float32) / 255.0

        return img

    # ------------------------------------------------------------------
    # Core fetch
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        try:
            curr = self.data[idx]
            next_ = self.data[idx + 1]   # ← target comes from t+1

            # -----------------------------
            # Load images (time t)
            # -----------------------------
            front = self._load_image(curr["image_front"])
            left  = self._load_image(curr["image_front_left"])
            right = self._load_image(curr["image_front_right"])
            seg   = self._load_image(curr["seg_front"], is_seg=True)

            # -----------------------------
            # Albumentations (CONSISTENT)
            # -----------------------------
            if self.transform is not None:
                augmented = self.transform(
                    image=front,
                    image_left=left,
                    image_right=right,
                    mask=seg
                )
                front = augmented["image"]
                left  = augmented["image_left"]
                right = augmented["image_right"]
                seg   = augmented["mask"]

            # -----------------------------
            # Convert to tensors
            # -----------------------------
            front = torch.from_numpy(front).permute(2, 0, 1).float()
            left  = torch.from_numpy(left).permute(2, 0, 1).float()
            right = torch.from_numpy(right).permute(2, 0, 1).float()

            seg = torch.from_numpy(seg).unsqueeze(0).float()  # (1, H, W)

            # -----------------------------
            # State vector (NO ACTIONS!)
            # -----------------------------
            state = torch.tensor([
                curr["velocity_x"],
                curr["velocity_y"],
                curr["velocity_z"],
                curr["speed_kmh"],
                curr["nearest_object_dist"],
                curr["box_count"]
            ], dtype=torch.float32)

            # -----------------------------
            # Target = NEXT timestep actions
            # -----------------------------
            target = torch.tensor([
                next_["steer"],
                next_["throttle"],
                next_["brake"]
            ], dtype=torch.float32)

            return {
                "front": front,     # (3, H, W)
                "left": left,       # (3, H, W)
                "right": right,     # (3, H, W)
                "seg": seg,         # (1, H, W)
                "state": state,     # (6,)
                "target": target    # (3,)
            }

        except Exception as e:
            logging.error(f"Error loading sample index {idx}", exc_info=True)
            raise CustomException(e, sys)
