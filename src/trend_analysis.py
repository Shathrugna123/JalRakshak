
from pathlib import Path
import csv
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = ROOT / "outputs" / "waste_analytics" / "frame_analytics.csv"
OUTPUT_CSV = ROOT / "outputs" / "waste_analytics" / "trend_summary.csv"

SEGMENT_SIZE = 30  # frames per segment


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Analytics CSV not found: {INPUT_CSV}")

    with INPUT_CSV.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print("No analytics data found.")
        return

    segments = defaultdict(list)

    for row in rows:
        frame = int(row["frame"])
        count = int(row["detections"])
        segment_start = (frame // SEGMENT_SIZE) * SEGMENT_SIZE
        segments[segment_start].append(count)

    summary = []

    for start, counts in sorted(segments.items()):
        summary.append({
            "start_frame": start,
            "end_frame": min(start + SEGMENT_SIZE - 1, len(rows) - 1),
            "frames_analyzed": len(counts),
            "total_detections": sum(counts),
            "average_detections": round(sum(counts) / len(counts), 2),
        })

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "start_frame", "end_frame", "frames_analyzed",
            "total_detections", "average_detections"
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    busiest = max(summary, key=lambda item: item["average_detections"])

    print("\n===== JALRAKSHAK TREND ANALYSIS =====")
    print("Frames analyzed:", len(rows))
    print("Segments analyzed:", len(summary))

    for item in summary:
        print(
            f"Frames {item['start_frame']}-{item['end_frame']}: "
            f"total={item['total_detections']}, "
            f"average={item['average_detections']}"
        )

    print(
        f"\nHighest-average segment: "
        f"frames {busiest['start_frame']}-{busiest['end_frame']} "
        f"(average {busiest['average_detections']} detections/frame)"
    )
    print("CSV saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
