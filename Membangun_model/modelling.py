import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===============================
# 1. Load Dataset
# ===============================
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

# ===============================
# 2. MLflow Setup
# ===============================
mlflow.set_experiment("Telco-Churn-Baseline")
mlflow.sklearn.autolog()

# ===============================
# 3. Training
# ===============================
with mlflow.start_run(run_name="RandomForest_Baseline"):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ===============================
    # 4. Evaluation
    # ===============================
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1)
