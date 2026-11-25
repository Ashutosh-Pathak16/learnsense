import time
import cv2
from dataset_recorder import DatasetRecorder
from feature_extractor import VisionPipeline


LABEL_NAMES = {
    None: "PAUSED (Press 0, 1, or 2)",
    0: "0: ATTENTIVE",
    1: "1: DISTRACTED",
    2: "2: SLEEPING / DROWSY",
}


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    pipeline = VisionPipeline(max_num_faces=3)
    recorder = DatasetRecorder("data/training_data.csv")
    total_logged = 0

    print("Controls: [0]=Attentive, [1]=Distracted, [2]=Sleeping, [n]=Pause, [q]=Quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            now = time.time()
            _, faces = pipeline.extract_features(frame, timestamp=now)

            logged_now = recorder.record_faces(faces)
            total_logged += logged_now

            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {face['track_id']} | EAR:{face['ear']:.2f} Yaw:{face['yaw']:.0f}",
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            status_text = f"Mode: {LABEL_NAMES[recorder.active_label]} | Rows Saved: {total_logged}"
            cv2.putText(frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(
                frame,
                "Keys: [0] Attentive  [1] Distracted  [2] Sleeping  [n] Pause  [q] Quit",
                (15, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow("LearnSense - Dataset Recorder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("0"):
                recorder.set_label(0)
            elif key == ord("1"):
                recorder.set_label(1)
            elif key == ord("2"):
                recorder.set_label(2)
            elif key == ord("n"):
                recorder.set_label(None)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()
        recorder.close()
        print(f"Finished recording. Total rows added this session: {total_logged}")
        print(f"Dataset file: {recorder.csv_path}")


if __name__ == "__main__":
    main()