
from pathlib import Path
import csv
import os

import numpy as np
import cv2
from ultralytics import YOLO


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "Floating Waste and River Vegetation Dataset"
IMAGE_DIR = DATASET / "val" / "images"
LABEL_DIR = DATASET / "val" / "labels"

MODEL_PATH = (
    ROOT / "outputs" / "jalrakshak_v2_10epochs"
    / "weights" / "best.pt"
)

CLASS_NAMES = ["floating_waste", "river_vegetation"]

# Set confidence from PowerShell environment variable.
# Default is 0.25.
CONFIDENCE = float(os.getenv("JALRAKSHAK_CONF", "0.25"))

IOU_MATCH_THRESHOLD = 0.50
INFERENCE_SIZE = 512
NUM_ERROR_IMAGES = 15

# Each threshold gets its own output folder.
CONF_LABEL = str(CONFIDENCE).replace(".", "_")
OUTPUT_DIR = ROOT / "outputs" / f"v2_error_analysis_conf_{CONF_LABEL}"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "examples"


# --------------------------------------------------
# 2. READ GROUND-TRUTH LABELS
# --------------------------------------------------

def read_ground_truth(label_path, image_width, image_height):
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
                    f"Invalid annotation in {label_path}, "
                    f"line {line_number}"
                )

            class_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            if class_id not in range(len(CLASS_NAMES)):
                raise ValueError(
                    f"Unknown class ID {class_id} in {label_path}"
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
# 3. CALCULATE IoU
# --------------------------------------------------

def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )
    area_b = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


# --------------------------------------------------
# 4. MATCH PREDICTIONS TO GROUND TRUTH
# --------------------------------------------------

def match_predictions(gt_boxes, gt_classes, pred_boxes, pred_classes):
    candidates = []

    for pred_idx, pred_box in enumerate(pred_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            if pred_classes[pred_idx] != gt_classes[gt_idx]:
                continue

            iou = calculate_iou(pred_box, gt_box)

            if iou >= IOU_MATCH_THRESHOLD:
                candidates.append((iou, pred_idx, gt_idx))

    # Match highest-IoU pairs first.
    candidates.sort(reverse=True)

    matched_predictions = set()
    matched_ground_truth = set()

    for iou, pred_idx, gt_idx in candidates:
        if pred_idx in matched_predictions:
            continue
        if gt_idx in matched_ground_truth:
            continue

        matched_predictions.add(pred_idx)
        matched_ground_truth.add(gt_idx)

    tp = len(matched_predictions)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_ground_truth)

    return tp, fp, fn, matched_predictions, matched_ground_truth


# --------------------------------------------------
# 5. DRAW VISUAL ERROR EXAMPLES
# --------------------------------------------------

def save_error_image(
    image,
    gt_boxes,
    gt_classes,
    pred_boxes,
    pred_classes,
    matched_predictions,
    matched_ground_truth,
    output_path,
):
    canvas = image.copy()
    image_height = canvas.shape[0]

    # Ground truth: green.
    for idx, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 200, 0)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"GT {CLASS_NAMES[gt_classes[idx]]}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

        if idx not in matched_ground_truth:
            cv2.putText(
                canvas,
                "MISSED",
                (x1, min(image_height - 5, y2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

    # Matched predictions: blue. Unmatched: red.
    for idx, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = map(int, box)

        if idx in matched_predictions:
            color = (255, 150, 0)
            status = "TP"
        else:
            color = (0, 0, 255)
            status = "UNMATCHED"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{status} {CLASS_NAMES[pred_classes[idx]]}",
            (x1, min(image_height - 5, y2 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), canvas):
        print(f"Warning: could not save {output_path}")


# --------------------------------------------------
# 6. MAIN EVALUATION
# --------------------------------------------------

def main():
    if not 0.0 <= CONFIDENCE <= 1.0:
        raise ValueError("Confidence must be between 0.0 and 1.0")

    for path in (IMAGE_DIR, LABEL_DIR, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    class_totals = {
        name: {"TP": 0, "FP": 0, "FN": 0}
        for name in CLASS_NAMES
    }

    image_results = []

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    print(f"Evaluating {len(image_paths)} validation images...")
    print(f"Model: {MODEL_PATH}")
    print(f"Confidence threshold: {CONFIDENCE}")
    print(f"IoU match threshold: {IOU_MATCH_THRESHOLD}")
    print(f"Inference image size: {INFERENCE_SIZE}")
    print(f"Output directory: {OUTPUT_DIR}")

    for image_idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        height, width = image.shape[:2]
        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        gt_boxes, gt_classes = read_ground_truth(
            label_path, width, height
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

        tp, fp, fn, _, _ = match_predictions(
            gt_boxes, gt_classes, pred_boxes, pred_classes
        )

        # Class-wise matching metrics.
        for class_id, class_name in enumerate(CLASS_NAMES):
            gt_indices = np.where(gt_classes == class_id)[0]
            pred_indices = np.where(pred_classes == class_id)[0]

            ctp, cfp, cfn, _, _ = match_predictions(
                gt_boxes[gt_indices],
                gt_classes[gt_indices],
                pred_boxes[pred_indices],
                pred_classes[pred_indices],
            )

            class_totals[class_name]["TP"] += ctp
            class_totals[class_name]["FP"] += cfp
            class_totals[class_name]["FN"] += cfn

        image_results.append({
            "image": image_path.name,
            "ground_truth": len(gt_boxes),
            "predictions": len(pred_boxes),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "confidence_mean": (
                round(float(pred_conf.mean()), 4)
                if len(pred_conf) else 0.0
            ),
            "missed_objects": fn,
            "unmatched_predictions": fp,
            "image_path": image_path,
        })

        if image_idx % 25 == 0:
            print(f"Processed {image_idx}/{len(image_paths)} images")

    # Save per-image report.
    csv_path = OUTPUT_DIR / "per_image_errors.csv"

    fields = [
        "image", "ground_truth", "predictions",
        "TP", "FP", "FN", "confidence_mean",
        "missed_objects", "unmatched_predictions",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for row in image_results:
            writer.writerow({key: row[key] for key in fields})

    # Save class summary.
    summary_path = OUTPUT_DIR / "class_error_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        fields = ["class", "TP", "FP", "FN", "precision", "recall"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for name, counts in class_totals.items():
            tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0

            writer.writerow({
                "class": name,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            })

    # Select images with the most unmatched objects/predictions.
    worst_images = sorted(
        image_results,
        key=lambda row: (row["FN"] + row["FP"], row["FN"]),
        reverse=True,
    )[:NUM_ERROR_IMAGES]

    print("\nSaving visual error examples...")

    for row in worst_images:
        image_path = row["image_path"]
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        gt_boxes, gt_classes = read_ground_truth(
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
        else:
            pred_boxes = np.empty((0, 4), dtype=float)
            pred_classes = np.empty((0,), dtype=int)

        _, _, _, matched_predictions, matched_ground_truth = (
            match_predictions(
                gt_boxes, gt_classes, pred_boxes, pred_classes
            )
        )

        save_error_image(
            image,
            gt_boxes,
            gt_classes,
            pred_boxes,
            pred_classes,
            matched_predictions,
            matched_ground_truth,
            IMAGE_OUTPUT_DIR / f"errors_{image_path.stem}.jpg",
        )

    print("\nERROR ANALYSIS COMPLETE")
    print(f"Per-image report: {csv_path}")
    print(f"Class summary:    {summary_path}")
    print(f"Visual examples:  {IMAGE_OUTPUT_DIR}")

    print("\nClass summary:")
    for name, counts in class_totals.items():
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        print(
            f"{name}: TP={tp}, FP={fp}, FN={fn}, "
            f"Precision={precision:.3f}, Recall={recall:.3f}"
        )


if __name__ == "__main__":
    main()
