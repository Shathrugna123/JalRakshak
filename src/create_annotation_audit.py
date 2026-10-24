
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "Floating Waste and River Vegetation Dataset"
IMAGE_DIRS = {
    "train": DATASET / "train" / "images",
    "val": DATASET / "val" / "images",
    "test": DATASET / "test" / "images",
}

LABEL_DIRS = {
    "train": DATASET / "train" / "labels",
    "val": DATASET / "val" / "labels",
    "test": DATASET / "test" / "labels",
}

OUTPUT_PATH = ROOT / "outputs" / "annotation_audit_checklist.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for split, image_dir in IMAGE_DIRS.items():
        label_dir = LABEL_DIRS[split]

        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(
                f"Missing image or label directory for split: {split}"
            )

        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                label_status = "MISSING_LABEL_FILE"
                waste_count = ""
                vegetation_count = ""
            else:
                label_status = "PRESENT"
                waste_count = 0
                vegetation_count = 0

                with label_path.open("r", encoding="utf-8") as file:
                    for line in file:
                        parts = line.strip().split()

                        if not parts:
                            continue

                        if len(parts) != 5:
                            label_status = "INVALID_LABEL_FORMAT"
                            continue

                        try:
                            class_id = int(parts[0])
                        except ValueError:
                            label_status = "INVALID_CLASS_ID"
                            continue

                        if class_id == 0:
                            waste_count += 1
                        elif class_id == 1:
                            vegetation_count += 1
                        else:
                            label_status = "UNKNOWN_CLASS_ID"

            rows.append({
                "split": split,
                "image": image_path.name,
                "label_file": label_path.name,
                "label_status": label_status,
                "floating_waste_boxes": waste_count,
                "river_vegetation_boxes": vegetation_count,
                "annotation_review": "NOT_REVIEWED",
                "missing_objects": "",
                "oversized_or_misplaced_boxes": "",
                "class_concerns": "",
                "notes": "",
            })

    fields = [
        "split",
        "image",
        "label_file",
        "label_status",
        "floating_waste_boxes",
        "river_vegetation_boxes",
        "annotation_review",
        "missing_objects",
        "oversized_or_misplaced_boxes",
        "class_concerns",
        "notes",
    ]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("ANNOTATION AUDIT CHECKLIST CREATED")
    print(f"Images listed: {len(rows)}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nReview the validation images first.")
    print("Do not modify original labels during this audit.")


if __name__ == "__main__":
    main()

