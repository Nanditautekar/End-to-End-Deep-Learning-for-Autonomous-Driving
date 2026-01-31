import os
import sys
import json
import cv2
import torch
import numpy as np

from src.Project.logger import logging
from src.Project.exception import CustomException
from src.Project.components.model_hfstream import SelfDrivingModel


class PredictionPipeline:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../")
        )

        self.checkpoint_path = checkpoint_path

        self.model = SelfDrivingModel().to(self.device)
        self._load_model()

        self.model.eval()

    # ---------------------------------------------------
    # Model loading
    # ---------------------------------------------------
    def _load_model(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )

        logging.info(f"Loaded model checkpoint from {self.checkpoint_path}")

    # ---------------------------------------------------
    # Image loading
    # ---------------------------------------------------
    def _load_rgb(self, rel_path):
        abs_path = os.path.join(
            self.project_root,
            rel_path.replace("\\", os.sep).replace("/", os.sep)
        )

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Missing image: {abs_path}")

        img = cv2.imread(abs_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0

        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(self.device)

    def _load_seg(self, rel_path):
        abs_path = os.path.join(
            self.project_root,
            rel_path.replace("\\", os.sep).replace("/", os.sep)
        )

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Missing seg image: {abs_path}")

        seg = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
        seg = cv2.resize(seg, (224, 224), interpolation=cv2.INTER_NEAREST)
        seg = torch.tensor(seg).unsqueeze(0).unsqueeze(0)

        return seg.to(self.device)

    # ---------------------------------------------------
    # MAIN PREDICTION FUNCTION
    # ---------------------------------------------------
    def predict(self, sample):
        """
        sample: dict with same keys as JSON row
        """

        try:
            # ---- Load inputs ----
            front = self._load_rgb(sample["image_front"])
            left  = self._load_rgb(sample["image_front_left"])
            right = self._load_rgb(sample["image_front_right"])
            seg   = self._load_seg(sample["seg_front"])

            state = torch.tensor([
                sample["velocity_x"],
                sample["velocity_y"],
                sample["velocity_z"],
                sample["speed_kmh"],
                sample["nearest_object_dist"],
                sample["box_count"]
            ], dtype=torch.float32).unsqueeze(0).to(self.device)

            # ---- Forward pass ----
            with torch.no_grad():
                preds = self.model(front, left, right, seg, state)

            steer = preds[0, 0].item()
            throttle = preds[0, 1].item()
            brake = preds[0, 2].item()

            # ---- SAFETY RULE (CRITICAL) ----
            if brake > 0.1:
                throttle = 0.0

            output = {
                "steer": float(steer),
                "throttle": float(throttle),
                "brake": float(brake)
            }

            return output

        except Exception as e:
            logging.error("Prediction failed", exc_info=True)
            raise CustomException(e, sys)
