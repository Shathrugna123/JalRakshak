
from pathlib import Path
import csv
from collections import Counter
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "outputs" / "jalrakshak_v2_10epochs" / "weights" / "best.pt"
VIDEO_PATH = ROOT / "videos" / "river_waste_test.mp4"
OUTPUT_DIR = ROOT / "outputs" / "waste_analytics"

CONF = 0.15
IMGSZ = 416


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(VIDEO_PATH),
        conf=CONF,
        imgsz=IMGSZ,
        stream=True,
        verbose=False,
    )

    rows = []
    class_counts = Counter()
    total_detections = 0

    for frame_number, result in enumerate(results):
        boxes = result.boxes
        counts = Counter()

        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.int().cpu().tolist()

            for class_id in class_ids:
                name = model.names[class_id]
                counts[name] += 1
                class_counts[name] += 1
                total_detections += 1

        rows.append({
            "frame": frame_number,
            "detections": sum(counts.values()),
            "floating_waste": counts.get("floating_waste", 0),
            "river_vegetation": counts.get("river_vegetation", 0),
        })

    csv_path = OUTPUT_DIR / "frame_analytics.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "frame", "detections", "floating_waste", "river_vegetation"
        ])
        writer.writeheader()
        writer.writerows(rows)

    frame_count = len(rows)
    frames_with_detections = sum(
        row["detections"] > 0 for row in rows
    )
    average = total_detections / frame_count if frame_count else 0

    busiest = sorted(
        rows, key=lambda row: row["detections"], reverse=True
    )[:5]

    print("\n===== JALRAKSHAK WASTE ANALYTICS =====")
    print("Frames analyzed:", frame_count)
    print("Total detections:", total_detections)
    print("Frames with detections:", frames_with_detections)
    print(f"Average detections per frame: {average:.2f}")
    print("Detections by class:", dict(class_counts))
    print("\nTop 5 frames by detection count:")

    for row in busiest:
        print(
            f"Frame {row['frame']}: "
            f"{row['detections']} detections"
        )

    print("\nCSV saved to:", csv_path)


if __name__ == "__main__":
    main()
