from pathlib import Path
from collections import Counter
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "outputs" / "jalrakshak_v2_10epochs" / "weights" / "best.pt"
VIDEO_PATH = ROOT / "videos" / "river_waste_test.mp4"
TRACKER_PATH = ROOT / "configs" / "bytetrack_jalrakshak.yaml"

model = YOLO(str(MODEL_PATH))

for conf in [0.10, 0.15]:
    print(f"\n===== CONFIDENCE: {conf} =====")

    results = model.track(
        source=str(VIDEO_PATH),
        tracker=str(TRACKER_PATH),
        imgsz=416,
        conf=conf,
        persist=False,
        stream=True,
        verbose=False,
    )

    total_detections = 0
    detections_with_ids = 0
    ids_seen = set()
    frames_with_ids = 0
    class_counts = Counter()

    for result in results:
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        total_detections += len(boxes)

        if boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            detections_with_ids += len(ids)
            ids_seen.update(ids)
            frames_with_ids += 1

        for class_id in boxes.cls.int().cpu().tolist():
            class_counts[model.names[class_id]] += 1

    print("Total detections:", total_detections)
    print("Detections with IDs:", detections_with_ids)
    print("Frames with IDs:", frames_with_ids)
    print("Unique track IDs:", len(ids_seen))
    print("Detections by class:", dict(class_counts))