
from pathlib import Path
import cv2
import yaml

ROOT = Path("data/Floating Waste and River Vegetation Dataset")
OUTPUT = Path("outputs/label_check.jpg")

def main():
    with (ROOT / "data.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config["names"]
    image_dir = ROOT / "train" / "images"
    label_dir = ROOT / "train" / "labels"

    # Find the first image with a non-empty label file.
    selected = None
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.is_file() and label_path.stat().st_size > 0:
            selected = (image_path, label_path)
            break

    if selected is None:
        print("No labeled training image found.")
        return

    image_path, label_path = selected
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path}")
        return

    height, width = image.shape[:2]
    valid_boxes = 0

    with label_path.open("r", encoding="utf-8") as f:
        rows = [line.split() for line in f if line.strip()]

    for row in rows:
        if len(row) != 5:
            print("Invalid label row:", row)
            continue

        try:
            class_id = int(row[0])
            xc, yc, bw, bh = map(float, row[1:])
        except ValueError:
            print("Invalid numeric label:", row)
            continue

        if not 0 <= class_id < len(names):
            print("Unknown class ID:", class_id)
            continue

        if not all(0 <= v <= 1 for v in (xc, yc, bw, bh)):
            print("Invalid normalized coordinates:", row)
            continue

        x1 = max(0, min(width - 1, int((xc - bw / 2) * width)))
        y1 = max(0, min(height - 1, int((yc - bh / 2) * height)))
        x2 = max(0, min(width - 1, int((xc + bw / 2) * width)))
        y2 = max(0, min(height - 1, int((yc + bh / 2) * height)))

        if x2 <= x1 or y2 <= y1:
            print("Invalid bounding box:", row)
            continue

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, names[class_id],
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        valid_boxes += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT), image)

    print("Image:", image_path.name)
    print("Resolution:", width, "x", height)
    print("Label rows:", len(rows))
    print("Valid boxes drawn:", valid_boxes)
    print("Saved to:", OUTPUT.resolve())

if __name__ == "__main__":
    main()
