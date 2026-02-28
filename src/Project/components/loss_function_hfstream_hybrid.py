import torch
import torch.nn as nn


class DrivingHybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):

        # Split predictions
        steer_pred = preds[:, 0]
        throttle_pred = preds[:, 1]
        brake_pred = preds[:, 2]

        # Split targets
        steer_true = targets[:, 0]
        throttle_true = targets[:, 1]
        brake_true = targets[:, 2]

        # Compute losses
        steer_loss = self.mse(steer_pred, steer_true)
        throttle_loss = self.mse(throttle_pred, throttle_true)
        brake_loss = self.bce(brake_pred, brake_true)

        total_loss = steer_loss + throttle_loss + brake_loss

        return total_loss