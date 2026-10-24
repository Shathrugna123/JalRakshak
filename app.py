
from pathlib import Path
import csv
import tempfile

import cv2
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    ROOT / "outputs" / "jalrakshak_v2_10epochs"
    / "weights" / "best.pt"
)
OUTPUT_DIR = ROOT / "outputs" / "app_results"

CONFIDENCE = 0.15
IMAGE_SIZE = 416

st.set_page_config(
    page_title="JalRakshak",
    page_icon="🌊",
    layout="wide",
)

st.title("JalRakshak")
st.subheader("AI-powered water waste monitoring")

st.write(
    "Upload a river video to detect floating waste and river vegetation."
)

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()


@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


model = load_model(str(MODEL_PATH))
st.success("JalRakshak V2 model loaded.")

uploaded_video = st.file_uploader(
    "Upload a river video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_video is not None:
    st.video(uploaded_video)

    if st.button("Run Detection", type="primary"):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        input_suffix = Path(uploaded_video.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=input_suffix
        ) as temp_file:
            temp_file.write(uploaded_video.getbuffer())
            input_path = Path(temp_file.name)

        output_path = OUTPUT_DIR / "annotated_video.mp4"
        csv_path = OUTPUT_DIR / "detection_report.csv"

        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps <= 0 or width <= 0 or height <= 0:
            input_path.unlink(missing_ok=True)
            st.error("Could not read the uploaded video's properties.")
            st.stop()

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            input_path.unlink(missing_ok=True)
            st.error("Could not create the annotated video.")
            st.stop()

        total_detections = 0
        class_counts = {}
        frame_rows = []
        progress = st.progress(0)
        status = st.empty()

        try:
            results = model.predict(
                source=str(input_path),
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                stream=True,
                verbose=False,
            )

            for frame_number, result in enumerate(results):
                annotated_frame = result.plot()
                writer.write(annotated_frame)

                counts = {}
                if result.boxes is not None and len(result.boxes) > 0:
                    for class_id in result.boxes.cls.int().cpu().tolist():
                        class_name = model.names[class_id]
                        counts[class_name] = counts.get(class_name, 0) + 1
                        class_counts[class_name] = (
                            class_counts.get(class_name, 0) + 1
                        )
                        total_detections += 1

                frame_rows.append({
                    "frame": frame_number,
                    "total_detections": sum(counts.values()),
                    "floating_waste": counts.get("floating_waste", 0),
                    "river_vegetation": counts.get("river_vegetation", 0),
                })

                if (frame_number + 1) % 10 == 0:
                    status.write(f"Processed {frame_number + 1} frames...")

        finally:
            writer.release()
            input_path.unlink(missing_ok=True)

        with csv_path.open("w", newline="", encoding="utf-8") as file:
            fields = [
                "frame", "total_detections",
                "floating_waste", "river_vegetation"
            ]
            csv_writer = csv.DictWriter(file, fieldnames=fields)
            csv_writer.writeheader()
            csv_writer.writerows(frame_rows)

        progress.progress(1.0)
        status.write("Detection complete.")

        st.session_state["jalrakshak_video"] = str(output_path)
        st.session_state["jalrakshak_csv"] = str(csv_path)
        st.session_state["jalrakshak_rows"] = frame_rows
        st.session_state["jalrakshak_total"] = total_detections
        st.session_state["jalrakshak_classes"] = class_counts

if "jalrakshak_video" in st.session_state:
    st.divider()
    st.header("Detection Results")

    col1, col2, col3 = st.columns(3)
    rows = st.session_state["jalrakshak_rows"]
    class_counts = st.session_state["jalrakshak_classes"]

    col1.metric("Frames analyzed", len(rows))
    col2.metric("Total detections", st.session_state["jalrakshak_total"])
    col3.metric(
        "Frames with detections",
        sum(row["total_detections"] > 0 for row in rows),
    )

    st.subheader("Detections by class")
    if class_counts:
        st.bar_chart(class_counts)
    else:
        st.info("No objects detected at the selected confidence threshold.")

    st.subheader("Annotated Video")
    video_path = Path(st.session_state["jalrakshak_video"])
    if video_path.exists():
        st.video(str(video_path))
        st.download_button(
            "Download annotated video",
            data=video_path.read_bytes(),
            file_name="jalrakshak_annotated.mp4",
            mime="video/mp4",
        )

    csv_path = Path(st.session_state["jalrakshak_csv"])
    if csv_path.exists():
        st.download_button(
            "Download detection CSV",
            data=csv_path.read_bytes(),
            file_name="jalrakshak_detection_report.csv",
            mime="text/csv",
        )
