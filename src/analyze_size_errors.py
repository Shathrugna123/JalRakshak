
from pathlib import Path
import csv

import numpy as np
import cv2
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

OUTPUT_DIR = ROOT / "outputs" / "v2_size_analysis_640"

CLASS_NAMES = ["floating_waste", "river_vegetation"]

CONFIDENCE = 0.15
IOU_MATCH_THRESHOLD = 0.50
INFERENCE_SIZE = 640

# Ground-truth box area as a percentage of original image area.
SMALL_MAX = 1.0
MEDIUM_MAX = 5.0

SIZE_CATEGORIES = ("small", "medium", "large")


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
                    f"Invalid class ID {class_id} in {label_path}"
                )

            # Convert normalized YOLO coordinates into pixels.
            x1 = (xc - w / 2) * image_width
            y1 = (yc - h / 2) * image_height
            x2 = (xc + w / 2) * image_width
            y2 = (yc + h / 2) * image_height

            x1 = max(0.0, min(float(image_width), x1))
            y1 = max(0.0, min(float(image_height), y1))
            x2 = max(0.0, min(float(image_width), x2))
            y2 = max(0.0, min(float(image_height), y2))

            boxes.append([x1, y1, x2, y2])
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

    area_a = max(0.0, box_a[2] - box_a[0]) * max(
        0.0, box_a[3] - box_a[1]
    )
    area_b = max(0.0, box_b[2] - box_b[0]) * max(
        0.0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# --------------------------------------------------
# 4. GREEDY ONE-TO-ONE MATCHING
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

    return matched_predictions, matched_ground_truth


# --------------------------------------------------
# 5. DETERMINE GROUND-TRUTH OBJECT SIZE
# --------------------------------------------------

def get_size_category(box, image_width, image_height):
    box_width = max(0.0, box[2] - box[0])
    box_height = max(0.0, box[3] - box[1])

    image_area = image_width * image_height
    box_area = box_width * box_height

    area_percentage = (
        100.0 * box_area / image_area if image_area else 0.0
    )

    if area_percentage < SMALL_MAX:
        category = "small"
    elif area_percentage < MEDIUM_MAX:
        category = "medium"
    else:
        category = "large"

    return category, area_percentage


# --------------------------------------------------
# 6. MAIN EVALUATION
# --------------------------------------------------

def main():
    if not 0.0 <= CONFIDENCE <= 1.0:
        raise ValueError("CONFIDENCE must be between 0 and 1")

    for path in (IMAGE_DIR, LABEL_DIR, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    size_totals = {
        class_name: {
            size: {"GT": 0, "TP": 0, "FN": 0}
            for size in SIZE_CATEGORIES
        }
        for class_name in CLASS_NAMES
    }

    class_totals = {
        class_name: {"TP": 0, "FP": 0, "FN": 0}
        for class_name in CLASS_NAMES
    }

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    print(f"Validation images: {len(image_paths)}")
    print(f"Model: {MODEL_PATH}")
    print(f"Confidence threshold: {CONFIDENCE}")
    print(f"IoU matching threshold: {IOU_MATCH_THRESHOLD}")
    print(f"Inference image size: {INFERENCE_SIZE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nAnalyzing object sizes...")

    per_object_rows = []
    processed = 0

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
        else:
            pred_boxes = np.empty((0, 4), dtype=float)
            pred_classes = np.empty((0,), dtype=int)

        # Match all classes together; only same-class matches are allowed.
        matched_predictions, matched_gt = match_predictions(
            gt_boxes, gt_classes, pred_boxes, pred_classes
        )

        # Calculate class-wise TP, FP, FN.
        for class_id, class_name in enumerate(CLASS_NAMES):
            gt_indices = np.where(gt_classes == class_id)[0]
            pred_indices = np.where(pred_classes == class_id)[0]

            class_gt_boxes = gt_boxes[gt_indices]
            class_gt_classes = gt_classes[gt_indices]
            class_pred_boxes = pred_boxes[pred_indices]
            class_pred_classes = pred_classes[pred_indices]

            class_matched_predictions, class_matched_gt = (
                match_predictions(
                    class_gt_boxes,
                    class_gt_classes,
                    class_pred_boxes,
                    class_pred_classes,
                )
            )

            tp = len(class_matched_predictions)
            fp = len(class_pred_boxes) - tp
            fn = len(class_gt_boxes) - len(class_matched_gt)

            class_totals[class_name]["TP"] += tp
            class_totals[class_name]["FP"] += fp
            class_totals[class_name]["FN"] += fn

        # Analyze each ground-truth object by its original image area.
        for gt_idx, gt_box in enumerate(gt_boxes):
            class_id = int(gt_classes[gt_idx])
            class_name = CLASS_NAMES[class_id]

            size_category, area_percentage = get_size_category(
                gt_box, width, height
            )

            detected = gt_idx in matched_gt

            counts = size_totals[class_name][size_category]
            counts["GT"] += 1
            counts["TP"] += int(detected)
            counts["FN"] += int(not detected)

            per_object_rows.append({
                "image": image_path.name,
                "class": class_name,
                "size": size_category,
                "box_area_percent": round(area_percentage, 5),
                "detected": int(detected),
            })

        processed += 1

        if image_idx % 25 == 0:
            print(f"Processed {image_idx}/{len(image_paths)} images")

    # --------------------------------------------------
    # 7. SAVE SIZE SUMMARY
    # --------------------------------------------------

    summary_path = OUTPUT_DIR / "size_error_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "class", "size", "GT", "TP", "FN",
            "recall", "miss_rate",
        ]

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for class_name in CLASS_NAMES:
            for size in SIZE_CATEGORIES:
                counts = size_totals[class_name][size]

                gt = counts["GT"]
                tp = counts["TP"]
                fn = counts["FN"]

                recall = tp / gt if gt else 0.0
                miss_rate = fn / gt if gt else 0.0

                writer.writerow({
                    "class": class_name,
                    "size": size,
                    "GT": gt,
                    "TP": tp,
                    "FN": fn,
                    "recall": round(recall, 4),
                    "miss_rate": round(miss_rate, 4),
                })

    # --------------------------------------------------
    # 8. SAVE PER-OBJECT REPORT
    # --------------------------------------------------

    object_path = OUTPUT_DIR / "per_object_size_errors.csv"

    with object_path.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "image", "class", "size",
            "box_area_percent", "detected",
        ]

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_object_rows)

    # --------------------------------------------------
    # 9. SAVE CLASS SUMMARY
    # --------------------------------------------------

    class_path = OUTPUT_DIR / "class_summary.csv"

    with class_path.open("w", newline="", encoding="utf-8") as file:
        fields = ["class", "TP", "FP", "FN", "precision", "recall"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for class_name, counts in class_totals.items():
            tp = counts["TP"]
            fp = counts["FP"]
            fn = counts["FN"]

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0

            writer.writerow({
                "class": class_name,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            })

    # --------------------------------------------------
    # 10. PRINT RESULTS
    # --------------------------------------------------

    print("\nSIZE ERROR ANALYSIS COMPLETE")
    print(f"Images processed: {processed}")
    print(f"Size summary: {summary_path}")
    print(f"Per-object report: {object_path}")
    print(f"Class summary: {class_path}")

    print("\nRecall by class and object size:")

    for class_name in CLASS_NAMES:
        print(f"\n{class_name}:")

        for size in SIZE_CATEGORIES:
            counts = size_totals[class_name][size]
            gt = counts["GT"]
            tp = counts["TP"]
            fn = counts["FN"]
            recall = tp / gt if gt else 0.0

            print(
                f"  {size:6s}: GT={gt:4d}, TP={tp:4d}, "
                f"FN={fn:4d}, Recall={recall:.3f}"
            )

    print("\nOverall class metrics:")

    for class_name, counts in class_totals.items():
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        print(
            f"{class_name}: TP={tp}, FP={fp}, FN={fn}, "
            f"Precision={precision:.3f}, Recall={recall:.3f}"
        )


if __name__ == "__main__":
    main()
