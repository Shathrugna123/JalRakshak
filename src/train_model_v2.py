
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = (
    PROJECT_ROOT
    / "data"
    / "Floating Waste and River Vegetation Dataset"
    / "data.yaml"
)

def main():
    if not DATASET_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATASET_YAML}"
        )

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Dataset:", DATASET_YAML)

    # Small pretrained model: suitable for an initial CPU experiment.
    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATASET_YAML),
        epochs=15,          # Short smoke test, not final training
        imgsz=416,         # Smaller images reduce CPU workload
        batch=4,           # Small batch to limit RAM usage
        device="cpu",
        workers=0,         # Safer on Windows
        cache=False,
        patience=5,
        project=str(PROJECT_ROOT / "outputs"),
        name="jalrakshak_v2_10epochs",
        exist_ok=True,
        plots=True,
    )

    print("\nTraining smoke test finished.")
    print("Check outputs/jalrakshak_v2_10epochs for results.")

if __name__ == "__main__":
    main()
