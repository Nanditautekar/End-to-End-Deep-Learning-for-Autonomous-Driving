import os
import sys
import torch
from torch.utils.data import DataLoader

import albumentations as A

from src.Project.logger import logging
from src.Project.exception import CustomException

from src.Project.components.dataset_class_hfstream import SelfDrivingDataset
from src.Project.components.model_hfstream import SelfDrivingModel
from src.Project.components.loss_function_hfstream import DrivingLoss
from src.Project.components.model_trainer import ModelTrainer


class TrainingPipeline:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../")
        )

        self.train_json = os.path.join(
            self.project_root, "data", "processed_json", "train.json"
        )
        self.val_json = os.path.join(
            self.project_root, "data", "processed_json", "val.json"
        )

        self.batch_size = 16
        self.epochs = 20
        self.lr = 1e-4

        self.ckpt_dir = os.path.join(self.project_root, "artifacts", "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def run(self):
        try:
            logging.info("Starting training pipeline")

            # -----------------------------
            # Albumentations (TRAIN ONLY)
            # -----------------------------
            train_tfms = A.Compose(
                [
                    A.RandomBrightnessContrast(p=0.3),
                    A.MotionBlur(p=0.2),
                    A.GaussianBlur(p=0.2),
                    A.HorizontalFlip(p=0.5),
                ],
                additional_targets={
                    "image_left": "image",
                    "image_right": "image",
                }
            )

            val_tfms = None

            # -----------------------------
            # Dataset & Loader
            # -----------------------------
            train_ds = SelfDrivingDataset(
                json_path=self.train_json,
                project_root=self.project_root,
                transform=train_tfms
            )

            val_ds = SelfDrivingDataset(
                json_path=self.val_json,
                project_root=self.project_root,
                transform=val_tfms
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,   # WINDOWS SAFE
                drop_last=True
            )

            val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )

            # -----------------------------
            # Model / Optim / Loss
            # -----------------------------
            model = SelfDrivingModel().to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            loss_fn = DrivingLoss().to(self.device)

            trainer = ModelTrainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=self.device
            )

            best_val_loss = float("inf")

            # -----------------------------
            # Training loop
            # -----------------------------
            for epoch in range(1, self.epochs + 1):
                train_loss = trainer.train_one_epoch(train_loader)
                val_loss = trainer.validate(val_loader)

                msg = (
                    f"Epoch [{epoch}/{self.epochs}] "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                )

                logging.info(msg)
                print(msg)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_path = os.path.join(self.ckpt_dir, "best_model.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    logging.info(f"Saved best model to {ckpt_path}")

            logging.info("Training completed successfully")

        except Exception as e:
            logging.error("Training pipeline failed", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    TrainingPipeline().run()
