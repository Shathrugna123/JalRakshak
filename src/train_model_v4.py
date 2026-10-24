
from pathlib import Path

import torch
from ultralytics import YOLO


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_YAML = (
    PROJECT_ROOT
    / "data"
    / "Floating Waste and River Vegetation Dataset"
    / "data.yaml"
)

# Start from V2's BEST checkpoint, not V3.
PREVIOUS_MODEL = (
    PROJECT_ROOT
    / "outputs"
    / "jalrakshak_v2_10epochs"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
RUN_NAME = "jalrakshak_v4_img512"


# --------------------------------------------------
# 2. TRAINING CONFIGURATION
# --------------------------------------------------

EPOCHS = 8
IMAGE_SIZE = 512
BATCH_SIZE = 4


def main():
    # Check required files
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATASET_YAML}"
        )

    if not PREVIOUS_MODEL.exists():
        raise FileNotFoundError(
            f"Previous model not found: {PREVIOUS_MODEL}"
        )

    # Display environment information
    print("=" * 60)
    print("JALRAKSHAK V4 - HIGHER RESOLUTION EXPERIMENT")
    print("=" * 60)

    print(f"Dataset YAML : {DATASET_YAML}")
    print(f"Starting model: {PREVIOUS_MODEL}")
    print(f"Output folder : {OUTPUT_DIR / RUN_NAME}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Image size    : {IMAGE_SIZE}")
    print(f"Batch size    : {BATCH_SIZE}")
    print("Device        : CPU")
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Load V2's best checkpoint
    model = YOLO(str(PREVIOUS_MODEL))

    # Train a new experiment
    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        workers=0,
        cache=False,
        patience=4,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        seed=42,
        pretrained=True,
        amp=False,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("V4 TRAINING FINISHED")
    print("=" * 60)

    print(f"Results folder: {OUTPUT_DIR / RUN_NAME}")
    print(
        "Best checkpoint:",
        OUTPUT_DIR / RUN_NAME / "weights" / "best.pt"
    )
    print(
        "Last checkpoint:",
        OUTPUT_DIR / RUN_NAME / "weights" / "last.pt"
    )

    print("\nNext step: compare V4 validation metrics against V2.")


if __name__ == "__main__":
    main()
