# 🌿 Plant Species Classifier

A machine learning pipeline that classifies plant species from morphological measurements.  
Built with **Python**, **Scikit-learn**, **Pandas**, and **MLflow** for experiment tracking.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11+ | Core language |
| Scikit-learn | Random Forest classifier |
| Pandas | Data loading & manipulation |
| MLflow | Experiment tracking & model logging |
| Joblib | Model serialization |

---

## Project Structure

```
plant-species-classifier/
├── src/
│   ├── explore.py      # Dataset overview and stats
│   ├── train.py        # Model training + MLflow logging
│   └── predict.py      # Load model and run inference
├── data/               # (place custom datasets here)
├── models/             # Saved model artifacts (git-ignored)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/luislhdev/plant-species-classifier.git
cd plant-species-classifier

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Explore the dataset
python src/explore.py

# 5. Train the model
python src/train.py

# 6. Run a prediction (sepal_length sepal_width petal_length petal_width)
python src/predict.py 5.1 3.5 1.4 0.2
```

---

## MLflow UI

After training, inspect your experiment runs:

```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

---

## Dataset

Uses the classic **Iris dataset** (Fisher, 1936) as a plant morphology proxy.  
150 samples across 3 species — Iris setosa, Iris versicolor, and Iris virginica —  
with 4 features: sepal length, sepal width, petal length, and petal width.

---

## Results

| Metric | Value |
|---|---|
| Algorithm | Random Forest (100 trees) |
| Test split | 20% |
| Accuracy | ~97% |

---

## Author

**Luis** — [@luislhdev](https://github.com/luislhdev)  
Part of an MLOps-focused portfolio. More projects coming soon.
