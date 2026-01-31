import os
import json
from src.Project.pipelines.prediction_pipeline import PredictionPipeline

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

CHECKPOINT = os.path.join(
    PROJECT_ROOT, "artifacts", "checkpoints", "best_model.pt"
)

TEST_JSON = os.path.join(
    PROJECT_ROOT, "data", "processed_json", "test.json"
)

# Load one sample
with open(TEST_JSON, "r") as f:
    sample = json.load(f)[0]

predictor = PredictionPipeline(CHECKPOINT)

output = predictor.predict(sample)

print("PREDICTED CONTROL:")
print(output)
