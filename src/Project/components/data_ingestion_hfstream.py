import logging
from datasets import load_dataset
from dataclasses import dataclass
from typing import Iterable, Dict, Any

logging.basicConfig(level=logging.INFO)


@dataclass
class DataIngestionConfig:
    dataset_name: str = "immanuelpeter/carla-autopilot-images"
    split: str = "train"
    streaming: bool = True


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def get_streaming_dataset(self) -> Iterable[Dict[str, Any]]:
        logging.info("Loading CARLA dataset in streaming mode")

        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.split,
            streaming=self.config.streaming
        )

        logging.info("Filtering required columns only")

        def select_columns(sample):
            return {
                "image_front": sample["image_front"],
                "image_front_left": sample["image_front_left"],
                "image_front_right": sample["image_front_right"],
                "speed_kmh": sample["speed_kmh"],
                "throttle": sample["throttle"],
                "steer": sample["steer"],
                "brake": sample["brake"]
            }

        dataset = dataset.map(select_columns)

        logging.info("Streaming dataset ready with selected columns only")
        return dataset


if __name__ == "__main__":
    config = DataIngestionConfig()
    ingestion = DataIngestion(config)
    dataset = ingestion.get_streaming_dataset()

    sample = next(iter(dataset))
    print("Available keys:", sample.keys())
