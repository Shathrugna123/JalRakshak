
from pathlib import Path

import cv2
from ultralytics import YOLO


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "Floating Waste and River Vegetation Dataset"
IMAGE_DIR = DATASET / "val" / "images"
LABEL_DIR = DATASET / "val" / "labels"

V2_PATH = (
    ROOT / "outputs" / "jalrakshak_v2_10epochs"
    / "weights" / "best.pt"
)

V4_PATH = (
    ROOT / "outputs" / "jalrakshak_v4_img512"
    / "weights" / "best.pt"
)

OUTPUT_DIR = ROOT / "outputs" / "ground_truth_v2_v4"

IMAGE_NAMES = [
    "FOTO_0039.jpg",
    "FOTO_0040.jpg",
    "FOTO_0027.jpg",
]

CLASS_NAMES = ["floating_waste", "river_vegetation"]

CONFIDENCE = 0.25
IOU_THRESHOLD = 0.7
INFERENCE_SIZE = 512


# --------------------------------------------------
# 2. DRAW GROUND-TRUTH LABELS
# --------------------------------------------------

def draw_ground_truth(image, label_path):
    height, width = image.shape[:2]
    annotated = image.copy()

    if not label_path.exists():
        print(f"WARNING: Label file missing: {label_path.name}")
        return annotated

    with label_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    for line_number, line in enumerate(lines, start=1):
        parts = line.strip().split()

        if not parts:
            continue

        if len(parts) != 5:
            print(f"Invalid annotation: {label_path.name}, line {line_number}")
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, box_width, box_height = map(
                float, parts[1:]
            )
        except ValueError:
            print(f"Unreadable annotation: {label_path.name}, line {line_number}")
            continue

        if class_id not in range(len(CLASS_NAMES)):
            print(f"Unknown class ID in {label_path.name}: {class_id}")
            continue

        # YOLO normalized coordinates -> image pixels
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        # Keep boxes within the image
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        label = CLASS_NAMES[class_id]

        # Green = dataset ground truth
        cv2.rectangle(
            annotated, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


# --------------------------------------------------
# 3. RUN MODEL PREDICTIONS
# --------------------------------------------------

def predict_image(model, image_path, title):
    result = model.predict(
        source=str(image_path),
        imgsz=INFERENCE_SIZE,
        conf=CONFIDENCE,
        iou=IOU_THRESHOLD,
        device="cpu",
        verbose=False,
    )[0]

    annotated = result.plot(
        labels=True,
        boxes=True,
        conf=True,
    )

    cv2.putText(
        annotated,
        title,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated, len(result.boxes)


# --------------------------------------------------
# 4. MAIN COMPARISON
# --------------------------------------------------

def main():
    for path in (IMAGE_DIR, LABEL_DIR, V2_PATH, V4_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading V2...")
    v2 = YOLO(str(V2_PATH))

    print("Loading V4...")
    v4 = YOLO(str(V4_PATH))

    for image_name in IMAGE_NAMES:
        image_path = IMAGE_DIR / image_name
        label_path = LABEL_DIR / f"{Path(image_name).stem}.txt"

        if not image_path.exists():
            print(f"Image not found, skipping: {image_path}")
            continue

        original = cv2.imread(str(image_path))

        if original is None:
            print(f"Could not read image: {image_path}")
            continue

        ground_truth = draw_ground_truth(original, label_path)

        v2_image, v2_count = predict_image(
            v2, image_path, "V2 - best model"
        )

        v4_image, v4_count = predict_image(
            v4, image_path, "V4 - 512px"
        )

        # Ensure all panels have identical dimensions
        target_height, target_width = ground_truth.shape[:2]

        v2_image = cv2.resize(v2_image, (target_width, target_height))
        v4_image = cv2.resize(v4_image, (target_width, target_height))

        comparison = cv2.hconcat([
            ground_truth,
            v2_image,
            v4_image,
        ])

        output_path = OUTPUT_DIR / (
            f"ground_truth_v2_v4_{Path(image_name).stem}.jpg"
        )

        if not cv2.imwrite(str(output_path), comparison):
            print(f"Failed to save: {output_path}")
            continue

        label_count = (
            sum(
                1 for line in label_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
            if label_path.exists()
            else 0
        )

        print(f"\nImage: {image_name}")
        print(f"Ground-truth boxes: {label_count}")
        print(f"V2 predictions:     {v2_count}")
        print(f"V4 predictions:     {v4_count}")
        print(f"Saved: {output_path}")

    print("\nComparison complete.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
