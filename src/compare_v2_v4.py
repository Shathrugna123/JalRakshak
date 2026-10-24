
from pathlib import Path

import cv2
from ultralytics import YOLO


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "Floating Waste and River Vegetation Dataset"
)

V2_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "jalrakshak_v2_10epochs"
    / "weights"
    / "best.pt"
)

V4_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "jalrakshak_v4_img512"
    / "weights"
    / "best.pt"
)

# Save comparisons here
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v2_v4_comparison"

# Number of images to compare
NUM_IMAGES = 12

# Use the same inference size for both models.
# This isolates the effect of their learned weights more fairly.
INFERENCE_SIZE = 512

CONFIDENCE = 0.25
IOU_THRESHOLD = 0.7

CLASS_NAMES = ["floating_waste", "river_vegetation"]


# --------------------------------------------------
# 2. DRAW PREDICTIONS
# --------------------------------------------------

def predict_and_draw(model, image_path, model_name):
    results = model.predict(
        source=str(image_path),
        imgsz=INFERENCE_SIZE,
        conf=CONFIDENCE,
        iou=IOU_THRESHOLD,
        device="cpu",
        verbose=False,
        save=False,
    )

    annotated = results[0].plot(
        labels=True,
        boxes=True,
        conf=True,
    )

    # Add a title to the image
    cv2.putText(
        annotated,
        model_name,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated, results[0]


# --------------------------------------------------
# 3. MAIN
# --------------------------------------------------

def main():
    if not V2_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"V2 model not found: {V2_MODEL_PATH}"
        )

    if not V4_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"V4 model not found: {V4_MODEL_PATH}"
        )

    val_images_dir = DATASET_DIR / "val" / "images"

    if not val_images_dir.exists():
        raise FileNotFoundError(
            f"Validation images not found: {val_images_dir}"
        )

    image_paths = sorted(
        path
        for path in val_images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not image_paths:
        raise RuntimeError("No validation images were found.")

    # Select a consistent set of images.
    image_paths = image_paths[:NUM_IMAGES]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading V2 model...")
    v2_model = YOLO(str(V2_MODEL_PATH))

    print("Loading V4 model...")
    v4_model = YOLO(str(V4_MODEL_PATH))

    print(f"\nComparing {len(image_paths)} validation images.")
    print(f"Output directory: {OUTPUT_DIR}")

    for index, image_path in enumerate(image_paths, start=1):
        print(
            f"[{index}/{len(image_paths)}] "
            f"Processing {image_path.name}"
        )

        original = cv2.imread(str(image_path))

        if original is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        v2_image, v2_result = predict_and_draw(
            v2_model, image_path, "V2 - best model"
        )

        v4_image, v4_result = predict_and_draw(
            v4_model, image_path, "V4 - 512px"
        )

        # Resize only for visual comparison if image dimensions differ.
        if v2_image.shape[:2] != v4_image.shape[:2]:
            v4_image = cv2.resize(
                v4_image,
                (v2_image.shape[1], v2_image.shape[0]),
            )

        # Put both predictions side by side.
        comparison = cv2.hconcat([v2_image, v4_image])

        output_path = OUTPUT_DIR / f"comparison_{image_path.stem}.jpg"

        success = cv2.imwrite(str(output_path), comparison)

        if not success:
            print(f"Could not save: {output_path}")
            continue

        v2_count = len(v2_result.boxes)
        v4_count = len(v4_result.boxes)

        print(
            f"  V2 detections: {v2_count} | "
            f"V4 detections: {v4_count}"
        )
        print(f"  Saved: {output_path.name}")

    print("\nComparison finished.")
    print(f"Open this folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
