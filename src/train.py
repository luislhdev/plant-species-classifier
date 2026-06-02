"""
Plant Species Classifier
Trains a Random Forest model on the Iris dataset as a plant species proxy.
Tracks experiments with MLflow.
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/plant_classifier.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and return features and labels as a DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # Map numeric targets to readable plant species names
    species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    labels = pd.Series(iris.target).map(species_map)
    return df, labels


def train(n_estimators: int = N_ESTIMATORS) -> None:
    """Train the classifier and log everything to MLflow."""
    print("Loading data...")
    X, y = load_data()

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # ── MLflow experiment tracking ─────────────────────────────────────────
    mlflow.set_experiment("plant-species-classifier")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        # Train
        model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(
                y_test, y_pred, target_names=le.classes_
            )
        )

        # Save model locally as well
        os.makedirs("models", exist_ok=True)
        joblib.dump({"model": model, "label_encoder": le}, MODEL_PATH)
        print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
