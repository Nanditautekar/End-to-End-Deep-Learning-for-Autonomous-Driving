import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Basic CNN block
# ------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# ------------------------------------------------------------
# Image Encoder (shared for front / left / right)
# ------------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)  # (B, 256)


# ------------------------------------------------------------
# Segmentation Encoder
# ------------------------------------------------------------
class SegmentationEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)  # (B, 128)


# ------------------------------------------------------------
# State Encoder
# ------------------------------------------------------------
class StateEncoder(nn.Module):
    def __init__(self, in_dim=6):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)  # (B, 128)


# ------------------------------------------------------------
# MAIN MODEL
# ------------------------------------------------------------
class SelfDrivingModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared encoders
        self.rgb_encoder = ImageEncoder(in_channels=3)
        self.seg_encoder = SegmentationEncoder()
        self.state_encoder = StateEncoder(in_dim=6)

        # Feature fusion size:
        # 3 * 256 (front/left/right)
        # + 128 (seg)
        # + 128 (state)
        fusion_dim = 256 * 3 + 128 + 128

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # -----------------------
        # Output heads
        # -----------------------

        # Steering (lane following)
        self.steer_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # Throttle
        self.throttle_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Brake (priority head)
        self.brake_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    # --------------------------------------------------------
    def forward(self, front, left, right, seg, state):
        """
        front, left, right: (B, 3, H, W)
        seg:                (B, 1, H, W)
        state:              (B, 6)
        """

        # Encode images
        f_feat = self.rgb_encoder(front)
        l_feat = self.rgb_encoder(left)
        r_feat = self.rgb_encoder(right)

        seg_feat = self.seg_encoder(seg)
        state_feat = self.state_encoder(state)

        # Fuse
        fused = torch.cat(
            [f_feat, l_feat, r_feat, seg_feat, state_feat],
            dim=1
        )

        x = self.trunk(fused)

        # Heads
        steer = self.steer_head(x)
        throttle = self.throttle_head(x)
        brake = self.brake_head(x)

        # Final output (B, 3)
        return torch.cat([steer, throttle, brake], dim=1)
