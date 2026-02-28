import sys
import torch
import torch.nn as nn
import torchvision.models as models

from src.Project.logger import logging
from src.Project.exception import CustomException


# ------------------------------------------------------------
# ResNet18 Image Encoder
# ------------------------------------------------------------
class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=False):
        try:
            super().__init__()
            logging.info("Initializing ResNet18 encoder")

            resnet = models.resnet18(pretrained=pretrained)

            # Removing final FC layer
            self.feature_extractor = nn.Sequential(
                *list(resnet.children())[:-1]
            )

            logging.info("ResNet18 encoder initialized successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def forward(self, x):
        try:
            logging.debug(f"ResNet input shape: {x.shape}")

            x = self.feature_extractor(x)
            x = x.view(x.size(0), -1)

            logging.debug(f"ResNet output shape: {x.shape}")

            return x

        except Exception as e:
            raise CustomException(e, sys)


# ------------------------------------------------------------
# Segmentation Encoder
# ------------------------------------------------------------
class SegmentationEncoder(nn.Module):
    def __init__(self):
        try:
            super().__init__()
            logging.info("Initializing Segmentation Encoder")

            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.AdaptiveAvgPool2d((1, 1))
            )

            logging.info("Segmentation Encoder initialized successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def forward(self, x):
        try:
            logging.debug(f"Seg input shape: {x.shape}")

            x = self.net(x)
            x = x.view(x.size(0), -1)

            logging.debug(f"Seg output shape: {x.shape}")

            return x

        except Exception as e:
            raise CustomException(e, sys)


# ------------------------------------------------------------
# State Encoder
# ------------------------------------------------------------
class StateEncoder(nn.Module):
    def __init__(self, in_dim=6):
        try:
            super().__init__()
            logging.info("Initializing State Encoder")

            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU()
            )

            logging.info("State Encoder initialized successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def forward(self, x):
        try:
            logging.debug(f"State input shape: {x.shape}")

            output = self.net(x)

            logging.debug(f"State output shape: {output.shape}")

            return output

        except Exception as e:
            raise CustomException(e, sys)


# ------------------------------------------------------------
# MAIN MODEL
# ------------------------------------------------------------
class SelfDrivingResNetModel(nn.Module):
    def __init__(self):
        try:
            super().__init__()
            logging.info("Initializing SelfDrivingResNetModel")

            self.rgb_encoder = ResNetEncoder(pretrained=False)
            self.seg_encoder = SegmentationEncoder()
            self.state_encoder = StateEncoder(in_dim=6)

            fusion_dim = 512 * 3 + 64 + 128

            self.trunk = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU()
            )

            self.steer_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh()
            )

            self.throttle_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

            self.brake_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

            logging.info("SelfDrivingResNetModel initialized successfully")

        except Exception as e:
            raise CustomException(e, sys)

    # --------------------------------------------------------
    def forward(self, front, left, right, seg, state):
        try:
            logging.debug("Starting forward pass")

            f_feat = self.rgb_encoder(front)
            l_feat = self.rgb_encoder(left)
            r_feat = self.rgb_encoder(right)

            seg_feat = self.seg_encoder(seg)
            state_feat = self.state_encoder(state)

            fused = torch.cat(
                [f_feat, l_feat, r_feat, seg_feat, state_feat],
                dim=1
            )

            logging.debug(f"Fused feature shape: {fused.shape}")

            x = self.trunk(fused)

            steer = self.steer_head(x)
            throttle = self.throttle_head(x)
            brake = self.brake_head(x)

            output = torch.cat([steer, throttle, brake], dim=1)

            logging.debug(f"Output shape: {output.shape}")
            logging.debug("Forward pass completed successfully")

            return output

        except Exception as e:
            raise CustomException(e, sys)
