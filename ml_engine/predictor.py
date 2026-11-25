import os
import warnings
import joblib
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None


STATE_LABELS = {
    0: "Attentive",
    1: "Distracted",
    2: "Sleeping / Drowsy",
}

FEATURE_ORDER = [
    "ear",
    "mar",
    "pitch",
    "yaw",
    "roll",
    "ear_mean_5s",
    "ear_std_5s",
    "abs_yaw_mean_5s",
]


class AttentionPredictor:
    def __init__(self, model_path="attention_model.joblib"):
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)

        self.model = None
        self.feature_columns = FEATURE_ORDER

        if os.path.exists(model_path):
            bundle = joblib.load(model_path)
            if isinstance(bundle, dict) and "model" in bundle:
                self.model = bundle["model"]
                self.feature_columns = bundle.get("feature_columns", FEATURE_ORDER)
            else:
                self.model = bundle

    @staticmethod
    def heuristic_predict(face_dict):
        ear = face_dict.get("ear", 0.3)
        mar = face_dict.get("mar", 0.1)
        pitch = face_dict.get("pitch", 0.0)
        yaw = face_dict.get("yaw", 0.0)
        ear_mean_5s = face_dict.get("ear_mean_5s", ear)

        if ear < 0.19 or ear_mean_5s < 0.20:
            code = 2
        elif abs(yaw) > 25.0 or pitch < -20.0 or mar > 0.65:
            code = 1
        else:
            code = 0
        return code, STATE_LABELS[code]

    def predict(self, face_dict):
        if self.model is None:
            return self.heuristic_predict(face_dict)

        values = [[float(face_dict.get(col, 0.0)) for col in self.feature_columns]]
        if pd is not None:
            x_input = pd.DataFrame(values, columns=self.feature_columns)
        else:
            x_input = np.array(values, dtype=np.float64)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            pred_code = int(self.model.predict(x_input)[0])

        return pred_code, STATE_LABELS.get(pred_code, "Attentive")