
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "outputs/jalrakshak_v2_10epochs/weights/best.pt"
VIDEO = ROOT / "videos/river_waste_test.mp4"

SAMPLE_FRAMES = [0, 25, 50, 56, 57, 75, 100, 117, 118, 125, 133, 140, 160]


def main():
    model = YOLO(str(MODEL))

    print("Frame | Class | Confidence | Track ID")
    print("-" * 52)

    for frame_index, result in enumerate(
        model.track(
            source=str(VIDEO),
            tracker="bytetrack.yaml",
            imgsz=416,
            conf=0.10,
            persist=True,
            stream=True,
            verbose=False,
        )
    ):
        if frame_index not in SAMPLE_FRAMES:
            continue

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            print(f"{frame_index:5d} | No detections")
            continue

        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            track_id = (
                int(boxes.id[i].item())
                if boxes.id is not None
                else None
            )

            print(
                f"{frame_index:5d} | "
                f"{model.names[class_id]:16s} | "
                f"{confidence:.3f} | "
                f"{track_id if track_id is not None else 'None'}"
            )


if __name__ == "__main__":
    main()
