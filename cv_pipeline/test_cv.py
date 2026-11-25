import time
import cv2
import numpy as np
from feature_extractor import VisionPipeline


def run_synthetic_check(pipeline):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _, faces = pipeline.extract_features(blank)
    summary = pipeline.summarize_frame(faces)
    assert faces == [], "Expected no faces on a blank frame"
    assert summary["total_faces"] == 0
    print("[OK] Synthetic blank-frame check passed.")


def run_webcam_check(pipeline):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[WARN] No webcam detected; skipping live preview.")
        return

    prev_time = time.time()
    print("[INFO] Live webcam check started. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            now = time.time()
            _, faces = pipeline.extract_features(frame, timestamp=now)
            summary = pipeline.summarize_frame(faces)

            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                code = face["state_code"]
                color = (0, 0, 255) if code == 2 else (0, 165, 255) if code == 1 else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID {face['track_id']}: {face['state_label']}",
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"EAR:{face['ear']:.2f} MAR:{face['mar']:.2f} Yaw:{face['yaw']:.0f} Pitch:{face['pitch']:.0f}",
                    (x1, min(frame.shape[0] - 10, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                )

            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
            prev_time = now
            cv2.putText(
                frame,
                f"FPS: {int(fps)} | Faces: {summary['total_faces']} | Score: {summary['avg_attention_score']}%",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

            cv2.imshow("LearnSense - Step 2 CV Check", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pipeline = VisionPipeline(max_num_faces=5)
    try:
        run_synthetic_check(pipeline)
        run_webcam_check(pipeline)
    finally:
        pipeline.close()