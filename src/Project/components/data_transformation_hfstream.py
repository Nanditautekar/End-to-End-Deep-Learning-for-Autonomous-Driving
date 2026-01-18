import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from datasets import load_dataset


class CarlaStreamingDataset(IterableDataset):
    def __init__(self, split="train"):
        self.dataset = load_dataset(
            "immanuelpeter/carla-autopilot-images",
            split=split,
            streaming=True
        )

        self.transform = transforms.Compose([
            transforms.Resize((66, 200)),
            transforms.ToTensor()
        ])

    def __iter__(self):
        for sample in self.dataset:
            front = self.transform(sample["image_front"])
            left = self.transform(sample["image_front_left"])
            right = self.transform(sample["image_front_right"])

            images = torch.cat([front, left, right], dim=0)  
            # shape → (9, 66, 200)

            speed = torch.tensor(sample["speed_kmh"], dtype=torch.float32)
            throttle = torch.tensor(sample["throttle"], dtype=torch.float32)
            steer = torch.tensor(sample["steer"], dtype=torch.float32)
            brake = torch.tensor(sample["brake"], dtype=torch.float32)

            control = torch.stack([speed, throttle, steer, brake])

            yield images, control
