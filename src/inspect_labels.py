
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    ROOT / "data" / "Floating Waste and River Vegetation Dataset"
    / "val" / "images" / "FOTO_0039.jpg"
)

LABEL_PATH = (
    ROOT / "data" / "Floating Waste and River Vegetation Dataset"
    / "val" / "labels" / "FOTO_0039.txt"
)

OUTPUT_PATH = ROOT / "outputs" / "FOTO_0039_ground_truth.jpg"

CLASS_NAMES = {
    0: "floating_waste",
    1: "river_vegetation",
}


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

    height, width = image.shape[:2]

    if not LABEL_PATH.exists():
        raise FileNotFoundError(f"Cannot read labels: {LABEL_PATH}")

    with LABEL_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.split()

            if not parts:
                continue

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid label on line {line_number}"
                )

            class_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])

            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)

            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            cv2.rectangle(
                image, (x1, y1), (x2, y2), (0, 255, 0), 2
            )

            class_name = CLASS_NAMES.get(
                class_id, f"unknown_class_{class_id}"
            )

            cv2.putText(
                image,
                f"{line_number}: {class_name}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(OUTPUT_PATH), image):
        raise RuntimeError("Failed to save output image")

    print(f"Ground-truth visualization saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
