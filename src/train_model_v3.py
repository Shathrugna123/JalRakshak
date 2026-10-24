
from pathlib import Path

import torch
from ultralytics import YOLO


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_YAML = (
    PROJECT_ROOT
    / "data"
    / "Floating Waste and River Vegetation Dataset"
    / "data.yaml"
)

PREVIOUS_MODEL = (
    PROJECT_ROOT
    / "outputs"
    / "jalrakshak_v2_10epochs"
    / "weights"
    / "last.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
RUN_NAME = "jalrakshak_v3_finetune"


def main():
    # Check required files
    if not DATASET_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATASET_YAML}"
        )

    if not PREVIOUS_MODEL.is_file():
        raise FileNotFoundError(
            f"Previous model checkpoint not found: {PREVIOUS_MODEL}"
        )

    # Display environment information
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Dataset:", DATASET_YAML)
    print("Starting checkpoint:", PREVIOUS_MODEL)

    # Load the weights from the previous training run
    model = YOLO(str(PREVIOUS_MODEL))

    # Continue fine-tuning with a new run directory
    model.train(
        data=str(DATASET_YAML),
        epochs=15,
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        cache=False,
        patience=8,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
    )

    print("\nFine-tuning completed.")
    print(f"Results: {OUTPUT_DIR / RUN_NAME}")
    print(
        "Best model:",
        OUTPUT_DIR / RUN_NAME / "weights" / "best.pt",
    )


if __name__ == "__main__":
    main()
                                                                                                                                                                                                                                              