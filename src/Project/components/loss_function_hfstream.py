import torch
import torch.nn as nn

class DrivingLoss(nn.Module):
    def __init__(
        self,
        steer_w=1.0,
        throttle_w=0.5,
        brake_w=3.0,
        conflict_w=2.0,
        brake_focus_threshold=0.05
    ):
        super().__init__()

        self.steer_w = steer_w
        self.throttle_w = throttle_w
        self.brake_w = brake_w
        self.conflict_w = conflict_w
        self.brake_focus_threshold = brake_focus_threshold

        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, preds, targets):
        """
        preds   : (B, 3) -> [steer, throttle, brake]
        targets : (B, 3)
        """

        steer_p, throttle_p, brake_p = preds[:, 0], preds[:, 1], preds[:, 2]
        steer_t, throttle_t, brake_t = targets[:, 0], targets[:, 1], targets[:, 2]

        # ----------------------------
        # 1. Steering loss
        # ----------------------------
        steer_loss = self.l1(steer_p, steer_t).mean()

        # ----------------------------
        # 2. Throttle loss
        # ----------------------------
        throttle_loss = self.l1(throttle_p, throttle_t).mean()

        # ----------------------------
        # 3. Brake loss (PRIORITY)
        # Extra weight when brake is actually required
        # ----------------------------
        brake_l1 = self.l1(brake_p, brake_t)

        brake_weight = torch.where(
            brake_t > self.brake_focus_threshold,
            torch.tensor(3.0, device=brake_t.device),
            torch.tensor(1.0, device=brake_t.device)
        )

        brake_loss = (brake_weight * brake_l1).mean()

        # ----------------------------
        # 4. Throttle–Brake conflict penalty
        # Penalize simultaneous activation
        # ----------------------------
        conflict_loss = torch.mean(throttle_p * brake_p)

        # ----------------------------
        # Total loss
        # ----------------------------
        total_loss = (
            self.steer_w * steer_loss +
            self.throttle_w * throttle_loss +
            self.brake_w * brake_loss +
            self.conflict_w * conflict_loss
        )

        return {
            "total": total_loss,
            "steer": steer_loss.detach(),
            "throttle": throttle_loss.detach(),
            "brake": brake_loss.detach(),
            "conflict": conflict_loss.detach()
        }
