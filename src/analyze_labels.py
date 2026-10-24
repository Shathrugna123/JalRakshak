
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "Floating Waste and River Vegetation Dataset"
SPLITS = ("train", "val", "test")
CLASS_NAMES = {0: "floating_waste", 1: "river_vegetation"}

# Collect normalized bounding-box areas by class.
areas = defaultdict(list)
counts = defaultdict(int)

for split in SPLITS:
    label_dir = DATASET / split / "labels"

    for label_file in label_dir.glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            _, _, width, height = map(float, parts[1:])
            area_percent = width * height * 100

            areas[class_id].append(area_percent)
            counts[class_id] += 1

print("Bounding-box area statistics (percentage of image area)")
for class_id, name in CLASS_NAMES.items():
    values = sorted(areas[class_id])

    if not values:
        print(f"{name}: no annotations found")
        continue

    median = values[len(values) // 2]
    small = sum(area < 1 for area in values)

    print(f"\n{name}")
    print(f"  Objects: {counts[class_id]}")
    print(f"  Median area: {median:.4f}%")
    print(f"  Smaller than 1% of image: {small} ({small / len(values):.1%})")

plt.figure(figsize=(9, 5))

for class_id, name in CLASS_NAMES.items():
    if areas[class_id]:
        plt.hist(
            areas[class_id],
            bins=50,
            alpha=0.6,
            label=name,
        )

plt.xlabel("Bounding-box area (% of image)")
plt.ylabel("Number of objects")
plt.title("JalRakshak: Bounding-box size distribution")
plt.legend()
plt.tight_layout()

output = ROOT / "outputs" / "bounding_box_sizes.png"
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=150)
plt.close()

print(f"\nChart saved to: {output}")
