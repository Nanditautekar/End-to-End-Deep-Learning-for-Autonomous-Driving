import torch
from model_hfstream import SelfDrivingModel

def main():
    model = SelfDrivingModel()

    B, H, W = 2, 224, 224

    front  = torch.randn(B, 3, H, W)
    left   = torch.randn(B, 3, H, W)
    right  = torch.randn(B, 3, H, W)
    seg    = torch.randint(0, 10, (B, 1, H, W)).float()
    state  = torch.randn(B, 6)

    out = model(front, left, right, seg, state)

    print("Output shape:", out.shape)
    print("Steer range:", out[:, 0].min().item(), out[:, 0].max().item())
    print("Throttle range:", out[:, 1].min().item(), out[:, 1].max().item())
    print("Brake range:", out[:, 2].min().item(), out[:, 2].max().item())

if __name__ == "__main__":
    main()


import torch
from loss_function_hfstream import DrivingLoss   # adjust import if filename differs

def sanity_check_loss():
    torch.manual_seed(0)

    B = 4  # batch size

    # ----------------------------
    # Fake predictions (model output)
    # steer in [-1,1], throttle/brake in [0,1]
    # ----------------------------
    preds = torch.tensor([
        [ 0.1, 0.8, 0.1],  # normal driving
        [-0.2, 0.6, 0.0],  # no brake
        [ 0.0, 0.9, 0.9],  # BAD: throttle + brake conflict
        [ 0.3, 0.2, 0.8],  # braking scenario
    ], dtype=torch.float32)

    # ----------------------------
    # Fake targets (ground truth)
    # ----------------------------
    targets = torch.tensor([
        [ 0.0, 0.7, 0.0],
        [-0.1, 0.5, 0.0],
        [ 0.0, 0.0, 1.0],  # should brake
        [ 0.2, 0.0, 1.0],
    ], dtype=torch.float32)

    loss_fn = DrivingLoss()

    losses = loss_fn(preds, targets)

    print("\n=== DRIVING LOSS SANITY CHECK ===")
    print(f"Total Loss    : {losses['total'].item():.4f}")
    print(f"Steer Loss    : {losses['steer'].item():.4f}")
    print(f"Throttle Loss : {losses['throttle'].item():.4f}")
    print(f"Brake Loss    : {losses['brake'].item():.4f}")
    print(f"Conflict Loss : {losses['conflict'].item():.4f}")

    # ----------------------------
    # Assertions (fail fast)
    # ----------------------------
    assert torch.isfinite(losses["total"]), "Total loss is NaN/Inf"
    assert losses["brake"] > 0, "Brake loss should be > 0"
    assert losses["conflict"] > 0, "Conflict loss should activate"

    print("✅ Loss sanity check PASSED")


if __name__ == "__main__":
    sanity_check_loss()

import os
import torch
from torch.utils.data import DataLoader

from src.Project.components.dataset_class_hfstream import SelfDrivingDataset
from src.Project.components.model_hfstream import SelfDrivingModel


def run_sanity_check():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../")
    )

    train_json = os.path.join(
        project_root, "data", "processed_json", "train.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SelfDrivingDataset(
        json_path=train_json,
        project_root=project_root
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    print("---- BATCH SHAPES ----")
    print("Front:", batch["front"].shape)    # [B, 3, 224, 224]
    print("Left:", batch["left"].shape)
    print("Right:", batch["right"].shape)
    print("Seg:", batch["seg"].shape)        # [B, 1, 224, 224]
    print("State:", batch["state"].shape)    # [B, N]
    print("Target:", batch["target"].shape)  # [B, 3]

    model = SelfDrivingModel().to(device)
    model.eval()

    with torch.no_grad():
        preds = model(
            batch["front"].to(device),
            batch["left"].to(device),
            batch["right"].to(device),
            batch["seg"].to(device),
            batch["state"].to(device)
        )

    print("\n---- MODEL OUTPUT ----")
    print(preds)
    print("Steer range:", preds[:, 0].min().item(), preds[:, 0].max().item())
    print("Throttle range:", preds[:, 1].min().item(), preds[:, 1].max().item())
    print("Brake range:", preds[:, 2].min().item(), preds[:, 2].max().item())


if __name__ == "__main__":
    run_sanity_check()

