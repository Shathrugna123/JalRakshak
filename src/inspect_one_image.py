
from pathlib import Path
import argparse
from PIL import Image, ImageDraw

ROOT = Path("data/Floating Waste and River Vegetation Dataset")

CLASS_NAMES = {
    0: "floating_waste",
    1: "river_vegetation",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_name", help="Image filename without extension")
    args = parser.parse_args()

    name = args.image_name
    image_path = ROOT / "val" / "images" / f"{name}.jpg"
    label_path = ROOT / "val" / "labels" / f"{name}.txt"
    output_path = Path("outputs") / f"{name}_ground_truth.jpg"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    with open(label_path, "r", encoding="utf-8") as file:
        labels = [line.strip() for line in file if line.strip()]

    for line in labels:
        values = line.split()
        if len(values) != 5:
            print("Skipping malformed label:", line)
            continue

        class_id = int(values[0])
        xc, yc, bw, bh = map(float, values[1:])

        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        color = "lime" if class_id == 0 else "cyan"
        label = CLASS_NAMES.get(class_id, f"class_{class_id}")

        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1, max(0, y1 - 18)), label, fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)

    print(f"Image: {image_path}")
    print(f"Annotations: {len(labels)}")
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
