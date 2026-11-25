import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "cv_pipeline") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "cv_pipeline"))

from cv_pipeline.feature_extractor import VisionPipeline
from ml_engine.predictor import AttentionPredictor


def post_json(url, payload, api_key="", timeout=3.0):
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Train-Key"] = api_key

    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def parse_args():
    parser = argparse.ArgumentParser(description="LearnSense Unified Application Runner")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--max-faces", type=int, default=5, help="Max faces to track")
    parser.add_argument("--subject", type=str, default="Live Lecture", help="Session subject name")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000", help="FastAPI base URL")
    parser.add_argument("--interval", type=float, default=1.5, help="Seconds between backend metric posts")
    parser.add_argument("--no-api", action="store_true", help="Run CV + ML locally without sending to API")
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("LEARNSENSE_API_KEY", ""),
        help="Access key sent in X-Train-Key header (or set LEARNSENSE_API_KEY)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    api_base = args.api_url.rstrip("/")
    session_id = None

    if not args.no_api:
        try:
            res = post_json(
                f"{api_base}/api/sessions/start",
                {"subject_name": args.subject},
                api_key=args.api_key,
            )
            session_id = res.get("session_id")
            print(f"[API] Started session #{session_id} ('{args.subject}')")
        except Exception as exc:
            print(f"[API Warn] Could not start backend session ({exc}). Continuing locally.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}.")

    pipeline = VisionPipeline(max_num_faces=args.max_faces)
    predictor = AttentionPredictor("attention_model.joblib")
    model_mode = "ML Model (attention_model.joblib)" if predictor.model is not None else "Heuristic Fallback"
    print(f"[ML] Classifier mode: {model_mode}. Press 'q' to stop.")

    prev_time = time.time()
    last_post_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            now = time.time()
            _, faces = pipeline.extract_features(frame, timestamp=now)

            for face in faces:
                code, label = predictor.predict(face)
                face["state_code"] = code
                face["state_label"] = label

                x1, y1, x2, y2 = face["bbox"]
                color = (0, 0, 255) if code == 2 else (0, 165, 255) if code == 1 else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID {face['track_id']}: {label}",
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

            summary = pipeline.summarize_frame(faces)

            if not args.no_api and session_id is not None and (now - last_post_time) >= args.interval:
                last_post_time = now
                payload = {
                    "session_id": session_id,
                    "total_faces": summary["total_faces"],
                    "attentive_count": summary["attentive_count"],
                    "distracted_count": summary["distracted_count"],
                    "sleeping_count": summary["sleeping_count"],
                    "avg_attention_score": summary["avg_attention_score"],
                }
                try:
                    post_json(f"{api_base}/api/metrics/log", payload, api_key=args.api_key)
                except (urllib.error.URLError, TimeoutError, Exception) as exc:
                    print(f"[API Warn] Failed to post metrics: {exc}")

            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
            prev_time = now

            hud = (
                f"FPS:{int(fps)} | Faces:{summary['total_faces']} | "
                f"Att:{summary['attentive_count']} Dis:{summary['distracted_count']} Slp:{summary['sleeping_count']} | "
                f"Score:{summary['avg_attention_score']}%"
            )
            cv2.putText(frame, hud, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("LearnSense - Live Monitor", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()
        if not args.no_api and session_id is not None:
            try:
                post_json(
                    f"{api_base}/api/sessions/{session_id}/end",
                    {},
                    api_key=args.api_key,
                )
                print(f"[API] Ended session #{session_id}")
            except Exception:
                pass


if __name__ == "__main__":
    main()