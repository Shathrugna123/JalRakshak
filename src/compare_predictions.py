
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "Floating Waste and River Vegetation Dataset"
IMAGE_PATH = DATASET / "test" / "images" / "FOTO_0002.jpg"
LABEL_PATH = DATASET / "test" / "labels" / "FOTO_0002.txt"
MODEL_PATH = ROOT / "outputs" / "jalrakshak_first_run" / "weights" / "best.pt"
OUTPUT_PATH = ROOT / "outputs" / "comparison_FOTO_0002.jpg"

CLASS_NAMES = {0: "floating_waste", 1: "river_vegetation"}


def draw_ground_truth(image):
    height, width = image.shape[:2]

    with LABEL_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.split()
            if len(values) != 5:
                continue

            class_id = int(values[0])
            xc, yc, bw, bh = map(float, values[1:])

            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                image,
                CLASS_NAMES[class_id],
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    return image


def main():
    for path in (IMAGE_PATH, LABEL_PATH, MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    original = cv2.imread(str(IMAGE_PATH))
    if original is None:
        raise ValueError(f"Could not read image: {IMAGE_PATH}")

    ground_truth = draw_ground_truth(original.copy())

    model = YOLO(str(MODEL_PATH))
    result = model.predict(
        source=str(IMAGE_PATH),
        imgsz=416,
        conf=0.25,
        device="cpu",
        verbose=False,
    )[0]

    predicted = result.plot()

    # Convert OpenCV BGR images to RGB for Matplotlib.
    ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)
    predicted = cv2.cvtColor(predicted, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(ground_truth)
    axes[0].set_title("Ground truth (green boxes)")
    axes[1].imshow(predicted)
    axes[1].set_title("YOLO predictions")
    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Comparison saved to: {OUTPUT_PATH}")
    print(f"Ground-truth labels: {sum(1 for _ in LABEL_PATH.open(encoding='utf-8'))}")
    print(f"Predictions at confidence >= 0.25: {len(result.boxes)}")


if __name__ == "__main__":
    main()
