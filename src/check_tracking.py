
from pathlib import Path
from collections import Counter, defaultdict
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "outputs/jalrakshak_v2_10epochs/weights/best.pt"
VIDEO_PATH = ROOT / "videos/river_waste_test.mp4"

def main():
    model = YOLO(str(MODEL_PATH))

    track_frames = defaultdict(set)
    track_classes = {}
    frames_with_ids = 0
    total_detections = 0
    detections_with_ids = 0
    previous_ids = set()
    consecutive_matches = 0
    frames = 0

    results = model.track(
        source=str(VIDEO_PATH),
        tracker="bytetrack.yaml",
        imgsz=416,
        conf=0.15,
        persist=True,
        stream=True,
        verbose=False,
    )

    for frame_index, result in enumerate(results):
        frames += 1
        current_ids = set()
        boxes = result.boxes

        if boxes is not None and len(boxes):
            total_detections += len(boxes)

            if boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                classes = boxes.cls.int().cpu().tolist()
                detections_with_ids += len(ids)
                frames_with_ids += 1

                for track_id, class_id in zip(ids, classes):
                    current_ids.add(track_id)
                    track_frames[track_id].add(frame_index)
                    track_classes[track_id] = model.names[class_id]

        consecutive_matches += len(previous_ids & current_ids)
        previous_ids = current_ids

    print("\n===== TRACK LIFETIME REPORT =====")
    print(f"Frames: {frames}")
    print(f"Total detections: {total_detections}")
    print(f"Detections with IDs: {detections_with_ids}")
    print(f"Frames containing IDs: {frames_with_ids}")
    print(f"Unique IDs: {len(track_frames)}")
    print(f"Consecutive-frame ID matches: {consecutive_matches}")

    for track_id, seen_frames in sorted(track_frames.items()):
        print(
            f"ID {track_id}: class={track_classes[track_id]}, "
            f"observed_frames={len(seen_frames)}, "
            f"first_frame={min(seen_frames)}, "
            f"last_frame={max(seen_frames)}"
        )

if __name__ == "__main__":
    main()
