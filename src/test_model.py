
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "jalrakshak_first_run"
    / "weights"
    / "best.pt"
)

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "Floating Waste and River Vegetation Dataset"
)

IMAGE_PATH = DATASET_DIR / "test" / "images" / "FOTO_0002.jpg"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Test image not found: {IMAGE_PATH}. "
            "Change IMAGE_PATH to an existing image in test/images."
        )

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(IMAGE_PATH),
        imgsz=416,
        conf=0.25,
        device="cpu",
        save=True,
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
    )

    for result in results:
        print("\nDetected objects:")
        if result.boxes is None or len(result.boxes) == 0:
            print("No objects detected at the selected confidence threshold.")
            continue

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = result.names[class_id]

            print(f"{class_name}: {confidence:.2%} confidence")

    print(f"\nPrediction images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
