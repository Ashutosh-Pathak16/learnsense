import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "ear",
    "mar",
    "pitch",
    "yaw",
    "roll",
    "ear_mean_5s",
    "ear_std_5s",
    "abs_yaw_mean_5s",
]
TARGET_COLUMN = "label"


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(project_root, "data", "training_data.csv")
    model_out = os.path.join(os.path.dirname(__file__), "attention_model.joblib")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training data not found at '{csv_path}'. Run record_data.py first.")

    df = pd.read_csv(csv_path).dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    if len(df) < 30:
        raise ValueError(f"Need at least 30 labeled rows to train; found {len(df)}.")

    print("Class distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    stratify_arg = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    joblib.dump(
        {"model": clf, "feature_columns": FEATURE_COLUMNS},
        model_out,
    )
    print(f"Saved trained model to: {model_out}")


if __name__ == "__main__":
    main()