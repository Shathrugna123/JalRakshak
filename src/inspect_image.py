
from pathlib import Path

import cv2


def inspect_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read this image.")
        return

    height, width, channels = image.shape

    print(f"Image: {image_path.name}")
    print(f"Width: {width} pixels")
    print(f"Height: {height} pixels")
    print(f"Channels: {channels}")
    print(f"File size: {image_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    print("JalRakshak - Image Inspection")
    inspect_image("data/sample.jpg")
