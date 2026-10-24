
from pathlib import Path
import csv
import statistics

ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = ROOT / "outputs" / "waste_analytics" / "frame_analytics.csv"
OUTPUT_CSV = ROOT / "outputs" / "waste_analytics" / "incident_alerts.csv"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Analytics CSV not found: {INPUT_CSV}")

    with INPUT_CSV.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print("No frame analytics found.")
        return

    counts = [int(row["detections"]) for row in rows]
    mean_count = statistics.mean(counts)
    std_count = statistics.pstdev(counts)

    # Flag frames that are unusually high compared with this video's baseline.
    threshold = mean_count + 2 * std_count

    alerts = []
    for row in rows:
        count = int(row["detections"])
        if count > threshold:
            alerts.append({
                "frame": row["frame"],
                "detections": count,
                "threshold": round(threshold, 2),
                "alert": "Review: unusually high detections",
            })

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        fields = ["frame", "detections", "threshold", "alert"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(alerts)

    print("\n===== JALRAKSHAK INCIDENT ALERTS =====")
    print("Frames analyzed:", len(rows))
    print(f"Average detections per frame: {mean_count:.2f}")
    print(f"Alert threshold: {threshold:.2f}")
    print("Potential alerts:", len(alerts))

    for alert in alerts:
        print(
            f"Frame {alert['frame']}: "
            f"{alert['detections']} detections — review recommended"
        )

    print("\nAlert report saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
