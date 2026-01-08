# =====================================================
# DAGSHUB + MLFLOW SETUP (ONLINE TRACKING)
# =====================================================
import dagshub
import mlflow
import mlflow.sklearn

dagshub.init(
    repo_owner="gesang-aja",
    repo_name="Telco-Churn-MLflow",
    mlflow=True
)

# =====================================================
# LIBRARIES
# =====================================================
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,precision_score,
    recall_score,f1_score,
    confusion_matrix,classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# =====================================================
# LOAD DATA (RELATIVE PATH – CI READY)
# =====================================================
DATA_PATH = "namadataset_prepocessing/TelcoCustomerChurn_prepocessing.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# MODEL + HYPERPARAMETER TUNING
# =====================================================
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

# =====================================================
# MLFLOW EXPERIMENT (DAGSHUB)
# =====================================================
mlflow.set_experiment("Telco-Churn-RF-DagsHub")

with mlflow.start_run(run_name="RandomForest_Tuning_Advanced"):

    # =========================
    # TRAINING
    # =========================
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # =========================
    # PREDICTION
    # =========================
    y_pred = best_model.predict(X_test)

    # =========================
    # METRICS (MANUAL LOGGING)
    # =========================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # =========================
    # EXTRA METRICS (ADVANCED)
    # =========================
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)

    mlflow.log_metric("specificity", specificity)
    mlflow.log_metric("false_positive", fp)
    mlflow.log_metric("false_negative", fn)

    # =========================
    # PARAMETERS
    # =========================
    mlflow.log_params({
        "model_type": "RandomForest",
        "n_estimators": best_model.n_estimators,
        "max_depth": best_model.max_depth,
        "min_samples_split": best_model.min_samples_split,
        "cv_folds": 3
    })

    # =========================
    # ARTIFACT 1: CONFUSION MATRIX
    # =========================
    os.makedirs("artifacts", exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    # =========================
    # ARTIFACT 2: CLASSIFICATION REPORT
    # =========================
    report = classification_report(y_test, y_pred, output_dict=True)
    report_path = "artifacts/classification_report.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    mlflow.log_artifact(report_path)

    # =========================
    # SAVE MODEL
    # =========================
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"
    )

    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", acc)
    print("F1-score:", f1)
