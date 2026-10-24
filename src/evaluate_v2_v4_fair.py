
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = (
    ROOT / "data"
    / "Floating Waste and River Vegetation Dataset"
    / "data.yaml"
)

V2 = (
    ROOT / "outputs" / "jalrakshak_v2_10epochs"
    / "weights" / "best.pt"
)

V4 = (
    ROOT / "outputs" / "jalrakshak_v4_img512"
    / "weights" / "best.pt"
)

OUTPUT = ROOT / "outputs" / "fair_v2_v4_evaluation"


def evaluate(model_path, run_name):
    print(f"\nEvaluating {run_name} at 512px...")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=512,
        batch=4,
        device="cpu",
        workers=0,
        plots=True,
        project=str(OUTPUT),
        name=run_name,
        exist_ok=True,
    )

    print(f"\n{run_name} results")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    for path in (V2, V4):
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

    evaluate(V2, "v2_val_img512")
    evaluate(V4, "v4_val_img512")

    print("\nBoth evaluations completed.")
    print(f"Results saved in: {OUTPUT}")


if __name__ == "__main__":
    main()
