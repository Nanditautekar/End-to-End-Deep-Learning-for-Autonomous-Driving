import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.Project.logger import logging
from src.Project.exception import CustomException
from src.Project.components.model_hfstream_ResNet import SelfDrivingResNetModel
from src.Project.components.loss_function_hfstream import DrivingLoss
from src.Project.components.dataset_class_hfstream import SelfDrivingDataset


class TrainingPipeline:
    def __init__(self):
        try:
            logging.info("Initializing Training Pipeline")

            self.batch_size = 4
            self.epochs = 20
            self.learning_rate = 1e-4
            self.model_save_path = "artifacts/best_resnet_model.pth"

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            logging.info(f"Using device: {self.device}")

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------
    def train(self, train_data_path, val_data_path):
        try:
            logging.info("Starting training process")

            # Dataset
            train_dataset = SelfDrivingDataset(train_data_path)
            val_dataset = SelfDrivingDataset(val_data_path)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )

            # Model
            model = SelfDrivingResNetModel().to(self.device)

            # Loss & Optimizer
            criterion = DrivingLoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate
            )

            best_val_loss = float("inf")

            # --------------------------------------------------------
            # Training Loop
            # --------------------------------------------------------
            for epoch in range(self.epochs):

                model.train()
                train_loss = 0.0

                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):

                    front, left, right, seg, state, target = batch

                    front = front.to(self.device)
                    left = left.to(self.device)
                    right = right.to(self.device)
                    seg = seg.to(self.device)
                    state = state.to(self.device)
                    target = target.to(self.device)

                    optimizer.zero_grad()

                    outputs = model(front, left, right, seg, state)

                    loss = criterion(outputs, target)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)

                # --------------------------------------------------------
                # Validation
                # --------------------------------------------------------
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch in val_loader:

                        front, left, right, seg, state, target = batch

                        front = front.to(self.device)
                        left = left.to(self.device)
                        right = right.to(self.device)
                        seg = seg.to(self.device)
                        state = state.to(self.device)
                        target = target.to(self.device)

                        outputs = model(front, left, right, seg, state)
                        loss = criterion(outputs, target)

                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)

                # Print Progress
                print(f"Epoch [{epoch+1}/{self.epochs}] "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f}")

                logging.info(f"Epoch {epoch+1}: "
                             f"Train Loss = {avg_train_loss:.4f}, "
                             f"Val Loss = {avg_val_loss:.4f}")

                # Save Best Model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    os.makedirs("artifacts", exist_ok=True)

                    torch.save(model.state_dict(), self.model_save_path)

                    logging.info("Best model saved successfully")

            logging.info("Training completed successfully")

        except Exception as e:
            raise CustomException(e, sys)


# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()

        pipeline.train(
            train_data_path="data/train.json",
            val_data_path="data/val.json"
        )

    except Exception as e:
        raise CustomException(e, sys)
