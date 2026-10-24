
from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLO


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "Floating Waste and River Vegetation Dataset"
IMAGE_DIR = DATASET / "val" / "images"
LABEL_DIR = DATASET / "val" / "labels"

MODEL_PATH = (
    ROOT / "outputs" / "jalrakshak_v2_10epochs"
    / "weights" / "best.pt"
)

OUTPUT_DIR = ROOT / "outputs" / "v2_prediction_audit"
VISUAL_DIR = OUTPUT_DIR / "examples"

CLASS_NAMES = ["floating_waste", "river_vegetation"]

CONFIDENCE = 0.15
MATCH_IOU = 0.50
INFERENCE_SIZE = 512
MAX_EXAMPLES = 30


# --------------------------------------------------
# 2. LOAD GROUND TRUTH
# --------------------------------------------------

def read_labels(label_path, image_width, image_height):
    boxes = []
    classes = []

    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid label in {label_path}, line {line_number}"
                )

            class_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            if class_id not in range(len(CLASS_NAMES)):
                raise ValueError(
                    f"Invalid class ID {class_id} in {label_path}"
                )

            x1 = (xc - w / 2) * image_width
            y1 = (yc - h / 2) * image_height
            x2 = (xc + w / 2) * image_width
            y2 = (yc + h / 2) * image_height

            boxes.append([
                max(0.0, min(float(image_width), x1)),
                max(0.0, min(float(image_height), y1)),
                max(0.0, min(float(image_width), x2)),
                max(0.0, min(float(image_height), y2)),
            ])
            classes.append(class_id)

    return (
        np.asarray(boxes, dtype=float).reshape(-1, 4),
        np.asarray(classes, dtype=int),
    )


# --------------------------------------------------
# 3. BOX IoU
# --------------------------------------------------

def calculate_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


# --------------------------------------------------
# 4. AUDIT MATCHING
# --------------------------------------------------

def audit_matches(gt_boxes, gt_classes, pred_boxes, pred_classes):
    """
    First match same-class boxes at the IoU threshold.
    Then inspect remaining predictions against unmatched ground truth
    to flag possible localization errors and class mismatches.

    These are audit categories, not automatic proof that a label is wrong.
    """

    candidates = []

    for pi, pred_box in enumerate(pred_boxes):
        for gi, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            candidates.append((iou, pi, gi))

    candidates.sort(reverse=True)

    used_preds = set()
    used_gt = set()
    records = []

    # First: same-class, sufficient-IoU matches.
    for iou, pi, gi in candidates:
        if iou < MATCH_IOU:
            break

        if pi in used_preds or gi in used_gt:
            continue

        if pred_classes[pi] != gt_classes[gi]:
            continue

        used_preds.add(pi)
        used_gt.add(gi)

        records.append({
            "prediction_index": pi,
            "ground_truth_index": gi,
            "status": "TP",
            "iou": iou,
        })

    # Second: inspect remaining predictions and labels.
    remaining_candidates = [
        item for item in candidates
        if item[1] not in used_preds and item[2] not in used_gt
    ]

    for iou, pi, gi in remaining_candidates:
        if pi in used_preds or gi in used_gt:
            continue

        same_class = pred_classes[pi] == gt_classes[gi]

        # High overlap, wrong class.
        if iou >= MATCH_IOU and not same_class:
            status = "CLASS_MISMATCH"
        # Some overlap, but insufficient IoU.
        elif iou >= 0.10 and same_class:
            status = "POSSIBLE_LOCALIZATION_ERROR"
        else:
            continue

        used_preds.add(pi)
        used_gt.add(gi)

        records.append({
            "prediction_index": pi,
            "ground_truth_index": gi,
            "status": status,
            "iou": iou,
        })

    # Predictions and labels that remain unmatched.
    for pi in range(len(pred_boxes)):
        if pi not in used_preds:
            records.append({
                "prediction_index": pi,
                "ground_truth_index": -1,
                "status": "UNMATCHED_PREDICTION",
                "iou": 0.0,
            })

    for gi in range(len(gt_boxes)):
        if gi not in used_gt:
            records.append({
                "prediction_index": -1,
                "ground_truth_index": gi,
                "status": "MISSED_GROUND_TRUTH",
                "iou": 0.0,
            })

    return records


# --------------------------------------------------
# 5. SAVE ANNOTATED VISUAL
# --------------------------------------------------

def save_visual(
    image, gt_boxes, gt_classes, pred_boxes, pred_classes,
    pred_conf, records, output_path
):
    canvas = image.copy()

    # Ground truth boxes: green.
    for gi, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            canvas,
            f"GT {CLASS_NAMES[gt_classes[gi]]}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 0),
            1,
        )

    # Prediction box colors are based on audit status.
    pred_status = {}

    for record in records:
        pi = record["prediction_index"]
        if pi >= 0:
            pred_status.setdefault(pi, record["status"])

    colors = {
        "TP": (255, 150, 0),
        "CLASS_MISMATCH": (255, 0, 255),
        "POSSIBLE_LOCALIZATION_ERROR": (0, 165, 255),
        "UNMATCHED_PREDICTION": (0, 0, 255),
    }

    for pi, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = map(int, box)
        status = pred_status.get(pi, "UNMATCHED_PREDICTION")
        color = colors.get(status, (0, 0, 255))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{status} {CLASS_NAMES[pred_classes[pi]]} "
            f"{pred_conf[pi]:.2f}",
            (x1, min(canvas.shape[0] - 5, y2 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


# --------------------------------------------------
# 6. RUN AUDIT
# --------------------------------------------------

def main():
    for path in (IMAGE_DIR, LABEL_DIR, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    audit_rows = []
    image_scores = []
    status_totals = {}

    print(f"Validation images: {len(image_paths)}")
    print(f"Confidence: {CONFIDENCE}")
    print(f"Matching IoU: {MATCH_IOU}")
    print(f"Inference size: {INFERENCE_SIZE}")

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        height, width = image.shape[:2]

        gt_boxes, gt_classes = read_labels(
            LABEL_DIR / f"{image_path.stem}.txt", width, height
        )

        result = model.predict(
            source=str(image_path),
            imgsz=INFERENCE_SIZE,
            conf=CONFIDENCE,
            iou=0.7,
            device="cpu",
            verbose=False,
        )[0]

        if result.boxes is not None and len(result.boxes) > 0:
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_classes = result.boxes.cls.cpu().numpy().astype(int)
            pred_conf = result.boxes.conf.cpu().numpy()
        else:
            pred_boxes = np.empty((0, 4), dtype=float)
            pred_classes = np.empty((0,), dtype=int)
            pred_conf = np.empty((0,), dtype=float)

        records = audit_matches(
            gt_boxes, gt_classes, pred_boxes, pred_classes
        )

        counts = {}
        for record in records:
            status = record["status"]
            counts[status] = counts.get(status, 0) + 1
            status_totals[status] = status_totals.get(status, 0) + 1

            pi = record["prediction_index"]
            gi = record["ground_truth_index"]

            audit_rows.append({
                "image": image_path.name,
                "status": status,
                "prediction_class": (
                    CLASS_NAMES[pred_classes[pi]] if pi >= 0 else ""
                ),
                "prediction_confidence": (
                    round(float(pred_conf[pi]), 4) if pi >= 0 else ""
                ),
                "ground_truth_class": (
                    CLASS_NAMES[gt_classes[gi]] if gi >= 0 else ""
                ),
                "IoU": round(float(record["iou"]), 4),
                "prediction_index": pi,
                "ground_truth_index": gi,
            })

        # Rank images with more possible class/localization errors and misses.
        priority = (
            counts.get("CLASS_MISMATCH", 0)
            + counts.get("POSSIBLE_LOCALIZATION_ERROR", 0)
            + counts.get("UNMATCHED_PREDICTION", 0)
            + counts.get("MISSED_GROUND_TRUTH", 0)
        )

        image_scores.append((priority, image_path.name, image_path))

        if index % 25 == 0:
            print(f"Processed {index}/{len(image_paths)} images")

    # Write detailed audit report.
    report_path = OUTPUT_DIR / "prediction_audit.csv"

    fields = [
        "image", "status", "prediction_class",
        "prediction_confidence", "ground_truth_class", "IoU",
        "prediction_index", "ground_truth_index",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    # Save visual examples of the most error-heavy images.
    worst_images = sorted(
        image_scores, key=lambda item: item[0], reverse=True
    )[:MAX_EXAMPLES]

    print("\nSaving visual audit examples...")

    for _, _, image_path in worst_images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]

        gt_boxes, gt_classes = read_labels(
            LABEL_DIR / f"{image_path.stem}.txt", width, height
        )

        result = model.predict(
            source=str(image_path),
            imgsz=INFERENCE_SIZE,
            conf=CONFIDENCE,
            iou=0.7,
            device="cpu",
            verbose=False,
        )[0]

        if result.boxes is not None and len(result.boxes) > 0:
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_classes = result.boxes.cls.cpu().numpy().astype(int)
            pred_conf = result.boxes.conf.cpu().numpy()
        else:
            pred_boxes = np.empty((0, 4), dtype=float)
            pred_classes = np.empty((0,), dtype=int)
            pred_conf = np.empty((0,), dtype=float)

        records = audit_matches(
            gt_boxes, gt_classes, pred_boxes, pred_classes
        )

        save_visual(
            image, gt_boxes, gt_classes, pred_boxes, pred_classes,
            pred_conf, records,
            VISUAL_DIR / f"audit_{image_path.stem}.jpg",
        )

    print("\nPREDICTION AUDIT COMPLETE")
    print(f"CSV report: {report_path}")
    print(f"Visual examples: {VISUAL_DIR}")
    print("\nAudit category totals:")

    for status, count in sorted(status_totals.items()):
        print(f"{status}: {count}")


if __name__ == "__main__":
    main()
