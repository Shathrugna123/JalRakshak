
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "videos" / "river_waste_test.mp4"
OUT = ROOT / "outputs" / "tracking_frame_inspection"

# Frame numbers selected around the tracking gap
FRAME_NUMBERS = [0, 25, 50, 56, 57, 75, 100, 117, 118, 125, 133, 140, 160]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO}")

    saved = 0

    for frame_number in FRAME_NUMBERS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()

        if not ok:
            print(f"Could not read frame {frame_number}")
            continue

        path = OUT / f"frame_{frame_number:03d}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"Saved: {path}")
        saved += 1

    cap.release()
    print(f"\nSaved {saved} inspection frames to: {OUT}")


if __name__ == "__main__":
    main()
