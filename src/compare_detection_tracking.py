
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "outputs/jalrakshak_v2_10epochs/weights/best.pt"
VIDEO = ROOT / "videos/river_waste_test.mp4"

SAMPLE_FRAMES = [0, 25, 50, 56, 57, 75, 100, 117, 118, 125, 133, 140, 160]


def main():
    model = YOLO(str(MODEL))

    detection_counts = Counter()
    tracking_counts = Counter()

    print("Frame | Detections | Tracked IDs")
    print("-" * 38)

    for result in model.track(
        source=str(VIDEO),
        tracker="bytetrack.yaml",
        imgsz=416,
        conf=0.15,
        persist=True,
        stream=True,
        verbose=False,
    ):
        frame_no = getattr(result, "frame", None)

        # Ultralytics may not expose the source frame index consistently;
        # count yielded frames instead.
        if not hasattr(main, "_frame_index"):
            main._frame_index = 0
        index = main._frame_index
        main._frame_index += 1

        if index not in SAMPLE_FRAMES:
            continue

        boxes = result.boxes
        detections = len(boxes) if boxes is not None else 0
        tracked = len(boxes.id) if boxes is not None and boxes.id is not None else 0

        detection_counts[index] = detections
        tracking_counts[index] = tracked

        print(f"{index:5d} | {detections:10d} | {tracked:11d}")

    print("\nSaved no files; this is a diagnostic only.")


if __name__ == "__main__":
    main()
