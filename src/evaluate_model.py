
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "Floating Waste and River Vegetation Dataset"
MODEL_PATH = ROOT / "outputs" / "jalrakshak_v2_10epochs" / "weights" / "best.pt"

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATASET / "data.yaml"),
        split="test",
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        plots=True,
        project=str(ROOT / "outputs"),
        name="baseline_test_evaluation",
        exist_ok=True,
    )

    print("\n=== JalRakshak Baseline Test Results ===")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"\nDetailed results: {ROOT / 'outputs' / 'baseline_test_evaluation'}")


if __name__ == "__main__":
    main()
