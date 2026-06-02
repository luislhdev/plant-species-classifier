"""
Plant Species Classifier — Prediction
Loads the trained model and predicts the species for given measurements.
"""

import joblib
import pandas as pd
import sys
import os

MODEL_PATH = "models/plant_classifier.pkl"

FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def predict(sepal_length: float, sepal_width: float,
            petal_length: float, petal_width: float) -> str:
    """Return the predicted species name for the given measurements."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run 'python src/train.py' first."
        )

    checkpoint = joblib.load(MODEL_PATH)
    model = checkpoint["model"]
    le = checkpoint["label_encoder"]

    sample = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=FEATURE_NAMES,
    )

    prediction = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]

    species = le.inverse_transform([prediction])[0]
    confidence = proba[prediction] * 100

    return species, confidence


if __name__ == "__main__":
    # Default sample: a typical Iris-setosa
    sepal_length = float(sys.argv[1]) if len(sys.argv) > 1 else 5.1
    sepal_width  = float(sys.argv[2]) if len(sys.argv) > 2 else 3.5
    petal_length = float(sys.argv[3]) if len(sys.argv) > 3 else 1.4
    petal_width  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2

    species, confidence = predict(sepal_length, sepal_width, petal_length, petal_width)

    print(f"\nMeasurements:")
    print(f"  Sepal: {sepal_length} cm (length) x {sepal_width} cm (width)")
    print(f"  Petal: {petal_length} cm (length) x {petal_width} cm (width)")
    print(f"\nPredicted Species : {species}")
    print(f"Confidence        : {confidence:.1f}%")
