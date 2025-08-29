# scripts/evaluation.py
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    print(f"\n📊 {model_name} Performance")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
    print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
    print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        try:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            print("ROC AUC (OVR):", roc_auc)
        except ValueError as e:
            print("ROC AUC could not be calculated:", e)
    else:
        print("Model does not support probability prediction for ROC AUC")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")

    os.makedirs("../results", exist_ok=True)
    plt.savefig(f"../results/{model_name}_confusion_matrix.png")
    plt.close()
