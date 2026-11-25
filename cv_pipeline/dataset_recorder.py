import csv
import os


class DatasetRecorder:
    FIELDNAMES = [
        "timestamp",
        "track_id",
        "ear",
        "mar",
        "pitch",
        "yaw",
        "roll",
        "ear_mean_5s",
        "ear_std_5s",
        "abs_yaw_mean_5s",
        "label",
    ]

    def __init__(self, csv_path="data/training_data.csv"):
        if not os.path.isabs(csv_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            csv_path = os.path.join(project_root, csv_path)

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_path = csv_path
        self.active_label = None  # None means paused; 0, 1, or 2 records that label

        file_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
        self._file = open(self.csv_path, mode="a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        if not file_exists:
            self._writer.writeheader()
            self._file.flush()

    def set_label(self, label):
        self.active_label = label

    def record_faces(self, faces_data):
        if self.active_label is None or not faces_data:
            return 0

        count = 0
        for face in faces_data:
            row = {
                "timestamp": round(face["timestamp"], 3),
                "track_id": face["track_id"],
                "ear": face["ear"],
                "mar": face["mar"],
                "pitch": face["pitch"],
                "yaw": face["yaw"],
                "roll": face["roll"],
                "ear_mean_5s": face["ear_mean_5s"],
                "ear_std_5s": face["ear_std_5s"],
                "abs_yaw_mean_5s": face["abs_yaw_mean_5s"],
                "label": self.active_label,
            }
            self._writer.writerow(row)
            count += 1

        self._file.flush()
        return count

    def close(self):
        self._file.close()