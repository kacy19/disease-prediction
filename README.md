# disease-prediction

# Disease Prediction 

## Overview

This project provides a complete toolkit to predict chest pain types from heart disease data. It covers data preprocessing, model training, evaluation, and visualization to support robust disease prediction.


## Project Structure

disease-prediction/
│
├── data/
│   └── heart\_dataset.csv            # Input dataset
│
├── results/                        # Folder for saving plots and evaluation outputs
│
├── scripts/
│   ├── preprocessing.py            # Data cleaning, encoding, scaling
│   ├── models.py                   # Model definitions (Logistic Regression, Random Forest, SVM)
│   ├── evaluation.py               # Model evaluation and confusion matrix plotting
│   ├── visualize.py                # ROC curves, feature importance visualizations
│   └── run\_pipeline.py             # Script to run the full pipeline end-to-end
│
├── tests/
│   └── test\_preprocessing.py       # Unit tests for preprocessing
│
├── requirements.txt
└── README.md


## Key Components

### 1. Data Preprocessing (`scripts/preprocessing.py`)

- Loads and cleans dataset.
- Combines chest pain indicator columns into one target.
- Encodes categorical variables.
- Scales features.
- Splits into train/test sets.

### 2. Models (`scripts/models.py`)

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

### 3. Evaluation (`scripts/evaluation.py`)

- Prints accuracy, precision, recall, F1-score.
- Computes ROC AUC (if supported).
- Saves confusion matrix plots in `results/`.

python
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    print(f"\n{model_name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
    print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
    print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        try:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            print("ROC AUC (OVR):", roc_auc)
        except ValueError:
            print("ROC AUC could not be calculated")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{model_name}_confusion_matrix.png")
    plt.close()


### 4. Running the Pipeline (`scripts/run_pipeline.py`)

python
from preprocessing import preprocess_data
from models import get_models
from evaluation import evaluate_model

(X_train, X_test, y_train, y_test), le_target = preprocess_data()
models = get_models()

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test, name)


## Visualizations

* Confusion matrices are automatically saved as PNG files inside the `results/` directory.
* Additional plots like ROC curves and feature importance can be generated using `scripts/visualize.py`.


## How to Run

1. Clone this repository.
2. Place `heart_dataset.csv` inside the `data/` folder.
3. Install dependencies:

bash
pip install -r requirements.txt

4. Run the full pipeline:

bash
python scripts/run_pipeline.py

5. Review model performance in the console and explore saved plots inside the `results/` directory.

## Testing

Run unit tests to validate data preprocessing:

bash
python -m unittest discover tests
