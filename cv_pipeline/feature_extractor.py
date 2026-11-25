import os
import time
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import distance as dist


class FaceTracker:
    """Assigns persistent track IDs across frames by matching face bounding-box centers."""

    def __init__(self, max_distance=120.0, max_missed_frames=10):
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.next_id = 1
        self.tracks = {}  # track_id -> {"center": (cx, cy), "missed": int}

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def assign_ids(self, bboxes):
        assigned_ids = [None] * len(bboxes)
        used_ids = set()

        for idx, bbox in enumerate(bboxes):
            cx, cy = self._center(bbox)
            best_id = None
            best_dist = self.max_distance

            for tid, info in self.tracks.items():
                if tid in used_ids:
                    continue
                tcx, tcy = info["center"]
                d = np.hypot(cx - tcx, cy - tcy)
                if d < best_dist:
                    best_dist = d
                    best_id = tid

            if best_id is not None:
                assigned_ids[idx] = best_id
                used_ids.add(best_id)
                self.tracks[best_id]["center"] = (cx, cy)
                self.tracks[best_id]["missed"] = 0
            else:
                new_id = self.next_id
                self.next_id += 1
                assigned_ids[idx] = new_id
                used_ids.add(new_id)
                self.tracks[new_id] = {"center": (cx, cy), "missed": 0}

        stale_ids = []
        for tid, info in self.tracks.items():
            if tid not in used_ids:
                info["missed"] += 1
                if info["missed"] > self.max_missed_frames:
                    stale_ids.append(tid)

        for tid in stale_ids:
            del self.tracks[tid]

        return assigned_ids


class VisionPipeline:
    def __init__(
        self,
        model_path="face_landmarker.task",
        max_num_faces=5,
        min_detection_conf=0.5,
        min_presence_conf=0.5,
        window_seconds=5.0,
    ):
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model asset not found at '{model_path}'. Download face_landmarker.task into cv_pipeline/."
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_conf,
            min_face_presence_confidence=min_presence_conf,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.tracker = FaceTracker()
        self.window_seconds = window_seconds
        self.history = {}  # track_id -> deque of (timestamp, ear, mar, pitch, yaw)

        # Landmark indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [61, 291, 13, 14]
        self.HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

        # Canonical 3D reference model points
        self.model_points_3d = np.array(
            [
                (0.0, 0.0, 0.0),           # Nose tip (1)
                (0.0, -330.0, -65.0),      # Chin (152)
                (-225.0, 170.0, -135.0),   # Left eye outer corner (33)
                (225.0, 170.0, -135.0),    # Right eye outer corner (263)
                (-150.0, -150.0, -125.0),  # Left mouth corner (61)
                (150.0, -150.0, -125.0),   # Right mouth corner (291)
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _calculate_ear(eye_points):
        v1 = dist.euclidean(eye_points[1], eye_points[5])
        v2 = dist.euclidean(eye_points[2], eye_points[4])
        h = dist.euclidean(eye_points[0], eye_points[3])
        return float((v1 + v2) / (2.0 * h)) if h > 1e-6 else 0.0

    @staticmethod
    def _calculate_mar(mouth_points):
        h = dist.euclidean(mouth_points[0], mouth_points[1])
        v = dist.euclidean(mouth_points[2], mouth_points[3])
        return float(v / h) if h > 1e-6 else 0.0

    def _estimate_head_pose(self, image_points_2d, img_w, img_h):
        focal_length = float(img_w)
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rot_vec, trans_vec = cv2.solvePnP(
            self.model_points_3d,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0, 0.0

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        proj_mat = np.hstack((rot_mat, trans_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_mat)

        pitch = float(euler_angles[0, 0])
        yaw = float(euler_angles[1, 0])
        roll = float(euler_angles[2, 0])
        return pitch, yaw, roll

    def _update_window_stats(self, track_id, now, ear, mar, pitch, yaw):
        if track_id not in self.history:
            self.history[track_id] = deque()

        dq = self.history[track_id]
        dq.append((now, ear, mar, pitch, yaw))

        cutoff = now - self.window_seconds
        while dq and dq[0][0] < cutoff:
            dq.popleft()

        ears = np.array([item[1] for item in dq], dtype=np.float64)
        yaws = np.abs(np.array([item[4] for item in dq], dtype=np.float64))

        return {
            "ear_mean_5s": round(float(np.mean(ears)), 4),
            "ear_std_5s": round(float(np.std(ears)), 4),
            "abs_yaw_mean_5s": round(float(np.mean(yaws)), 2),
        }

    @staticmethod
    def classify_heuristic(ear, mar, pitch, yaw, ear_mean_5s):
        """Baseline heuristic: 0=Attentive, 1=Distracted, 2=Sleeping/Drowsy."""
        if ear < 0.19 or ear_mean_5s < 0.20:
            return 2, "Sleeping / Drowsy"
        if abs(yaw) > 25.0 or pitch < -20.0 or mar > 0.65:
            return 1, "Distracted"
        return 0, "Attentive"

    def extract_features(self, frame, timestamp=None):
        if timestamp is None:
            timestamp = time.time()

        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        raw_items = []
        bboxes = []

        if detection_result.face_landmarks:
            for landmarks in detection_result.face_landmarks:
                landmarks_px = np.array(
                    [[int(lm.x * img_w), int(lm.y * img_h), lm.z * img_w] for lm in landmarks]
                )

                left_ear = self._calculate_ear(landmarks_px[self.LEFT_EYE, :2])
                right_ear = self._calculate_ear(landmarks_px[self.RIGHT_EYE, :2])
                avg_ear = (left_ear + right_ear) / 2.0
                mar = self._calculate_mar(landmarks_px[self.MOUTH, :2])

                head_pts_2d = landmarks_px[self.HEAD_POSE_LANDMARKS, :2].astype(np.float64)
                pitch, yaw, roll = self._estimate_head_pose(head_pts_2d, img_w, img_h)

                x_min, y_min = np.min(landmarks_px[:, :2], axis=0)
                x_max, y_max = np.max(landmarks_px[:, :2], axis=0)
                bbox = (
                    max(0, int(x_min)),
                    max(0, int(y_min)),
                    min(img_w - 1, int(x_max)),
                    min(img_h - 1, int(y_max)),
                )

                bboxes.append(bbox)
                raw_items.append((bbox, avg_ear, mar, pitch, yaw, roll))

        track_ids = self.tracker.assign_ids(bboxes)
        faces_data = []

        for tid, (bbox, avg_ear, mar, pitch, yaw, roll) in zip(track_ids, raw_items):
            win_stats = self._update_window_stats(tid, timestamp, avg_ear, mar, pitch, yaw)
            state_code, state_label = self.classify_heuristic(
                avg_ear, mar, pitch, yaw, win_stats["ear_mean_5s"]
            )

            faces_data.append(
                {
                    "track_id": tid,
                    "timestamp": timestamp,
                    "bbox": bbox,
                    "ear": round(avg_ear, 4),
                    "mar": round(mar, 4),
                    "pitch": round(pitch, 2),
                    "yaw": round(yaw, 2),
                    "roll": round(roll, 2),
                    "ear_mean_5s": win_stats["ear_mean_5s"],
                    "ear_std_5s": win_stats["ear_std_5s"],
                    "abs_yaw_mean_5s": win_stats["abs_yaw_mean_5s"],
                    "state_code": state_code,
                    "state_label": state_label,
                }
            )

        return detection_result, faces_data

    @staticmethod
    def summarize_frame(faces_data):
        total = len(faces_data)
        attentive = sum(1 for f in faces_data if f["state_code"] == 0)
        distracted = sum(1 for f in faces_data if f["state_code"] == 1)
        sleeping = sum(1 for f in faces_data if f["state_code"] == 2)
        score = round((attentive / total) * 100.0, 2) if total > 0 else 0.0
        return {
            "total_faces": total,
            "attentive_count": attentive,
            "distracted_count": distracted,
            "sleeping_count": sleeping,
            "avg_attention_score": score,
        }

    def close(self):
        self.detector.close()