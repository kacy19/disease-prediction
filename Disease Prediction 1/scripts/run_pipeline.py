# scripts/run_pipeline.py
from preprocessing import preprocess_data
from models import get_models
from evaluation import evaluate_model

(X_train, X_test, y_train, y_test), le_target = preprocess_data()

models = get_models()

for name, model in models.items():
    print(f"\n⏳ Training {name}...")
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test, name)
