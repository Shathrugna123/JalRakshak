
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import cv2

# Project paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "outputs" / "jalrakshak_v2_10epochs" / "weights" / "best.pt"
VIDEO_PATH = ROOT / "videos" / "river_waste_test.mp4"
OUTPUT_DIR = ROOT / "outputs" / "bytetrack_test"
OUTPUT_VIDEO = OUTPUT_DIR / "river_waste_tracked.mp4"

# Start with conservative settings for a CPU-only laptop
IMGSZ = 416
CONF = 0.15

CLASS_NAMES = {
    0: "floating_waste",
    1: "river_vegetation",
}


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    # Persistent tracking associates detections between video frames.
    results = model.track(
        source=str(VIDEO_PATH),
        tracker="configs/bytetrack_jalrakshak.yaml",
        imgsz=IMGSZ,
        conf=CONF,
        persist=True,
        stream=True,
        verbose=False,
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Could not read the input video's properties.")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create the output video.")

    track_classes = defaultdict(set)
    frames_processed = 0

    try:
        for result in results:
            frame = result.orig_img.copy()
            frames_processed += 1

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                class_ids = boxes.cls.cpu().numpy().astype(int)

                track_ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.id is not None
                    else [None] * len(boxes)
                )

                for box, class_id, track_id in zip(
                    xyxy, class_ids, track_ids
                ):
                    x1, y1, x2, y2 = box
                    class_name = CLASS_NAMES.get(
                        class_id, f"class_{class_id}"
                    )

                    if track_id is not None:
                        track_classes[class_name].add(track_id)
                        label = f"{class_name} ID:{track_id}"
                    else:
                        label = f"{class_name} ID:untracked"

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            writer.write(frame)

            if frames_processed % 25 == 0:
                print(f"Processed {frames_processed} frames")

    finally:
        writer.release()

    print("\nTracking finished.")
    print(f"Frames processed: {frames_processed}")
    print(f"Input FPS: {fps:.2f}")
    print(f"Output video: {OUTPUT_VIDEO}")

    print("\nUnique track IDs observed by class:")
    for class_name in CLASS_NAMES.values():
        print(f"{class_name}: {len(track_classes[class_name])}")


if __name__ == "__main__":
    main()
