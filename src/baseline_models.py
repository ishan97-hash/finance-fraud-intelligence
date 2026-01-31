import os
import json  # <-- NEW: to save feature list
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# make sure models folder exists (one level up from src/)
os.makedirs("../models", exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

# ---------- helpers ---------- #

def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("../models", exist_ok=True)  # <-- ensure top-level models too

def load_data(path):
    print(f"🔹 Loading cleaned data from {path}")
    df = pd.read_csv(path)
    print("✅ Loaded:", df.shape)
    return df

def build_feature_matrix(df, top_n=30):
    """
    Use correlation with isFraud to pick top_n features.
    """
    print("🔹 Building feature matrix (top correlated features)...")

    # isFraud is scaled, so we reconstruct binary target
    y = (df["isFraud"] > 0).astype(int)

    # Compute correlations with isFraud (absolute value)
    corr = df.corr(numeric_only=True)["isFraud"].abs().sort_values(ascending=False)

    # First entry is isFraud itself, skip it
    feature_names = corr.index[1: top_n + 1].tolist()

    print(f"🔹 Top {top_n} features used for modeling:")
    for f in feature_names:
        print("   -", f)

    X = df[feature_names].copy()
    return X, y, feature_names

def train_test_smote_split(X, y, test_size=0.2, random_state=42):
    print("🔹 Train-test split with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print("   Train shape:", X_train.shape, "Fraud rate:", y_train.mean())
    print("   Test  shape:", X_test.shape, "Fraud rate:", y_test.mean())

    print("🔹 Applying SMOTE on training data...")
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("   After SMOTE - Train shape:", X_train_res.shape, "Fraud rate:", y_train_res.mean())

    return X_train_res, X_test, y_train_res, y_test

def plot_confusion_matrix(cm, labels, title, file_path):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"✅ Saved confusion matrix to {file_path}")

# ---------- main pipeline ---------- #

def run_baseline_models():
    ensure_dirs()

    # 1. Load data
    df = load_data(r"S:\Study\INST_737_Data Science\Finance_Fraud_Project\data\processed\train_clean.csv")
    
    # 2. Build X, y
    X, y, feature_names = build_feature_matrix(df, top_n=30)

    # 🔹 NEW: save the exact feature list for SHAP later
    feature_list_path = "../models/feature_list.json"
    with open(feature_list_path, "w") as f:
        json.dump(feature_names, f)
    print(f"✅ Saved feature list to {feature_list_path}")

    # 3. Split + SMOTE
    X_train, X_test, y_train, y_test = train_test_smote_split(X, y)

    # 4. Logistic Regression
    print("\n===== Logistic Regression =====")
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    y_proba_lr = log_reg.predict_proba(X_test)[:, 1]

    print("🔹 Classification report (Logistic Regression):")
    print(classification_report(y_test, y_pred_lr, digits=4))

    auc_lr = roc_auc_score(y_test, y_proba_lr)
    print("🔹 ROC-AUC (LogReg):", auc_lr)

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    plot_confusion_matrix(
        cm_lr,
        labels=["Non-Fraud", "Fraud"],
        title="LogReg Confusion Matrix",
        file_path="outputs/figures/logreg_confusion.png"
    )

    RocCurveDisplay.from_predictions(y_test, y_proba_lr)
    plt.title("LogReg ROC Curve")
    plt.tight_layout()
    plt.savefig("outputs/figures/logreg_roc.png")
    plt.close()
    print("✅ Saved LogReg ROC curve.")

    # 5. Random Forest
    print("\n===== Random Forest =====")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    print("🔹 Classification report (Random Forest):")
    print(classification_report(y_test, y_pred_rf, digits=4))

    auc_rf = roc_auc_score(y_test, y_proba_rf)
    print("🔹 ROC-AUC (Random Forest):", auc_rf)

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plot_confusion_matrix(
        cm_rf,
        labels=["Non-Fraud", "Fraud"],
        title="Random Forest Confusion Matrix",
        file_path="outputs/figures/rf_confusion.png"
    )

    # 🔹 Save RF model for SHAP & later use
    joblib.dump(rf, "../models/random_forest.pkl")
    print("✅ Random Forest model saved to ../models/random_forest.pkl")

    RocCurveDisplay.from_predictions(y_test, y_proba_rf)
    plt.title("Random Forest ROC Curve")
    plt.tight_layout()
    plt.savefig("outputs/figures/rf_roc.png")
    plt.close()
    print("✅ Saved Random Forest ROC curve.")

    # 6. Save some metadata / summary
    summary_path = "outputs/model_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Baseline Models Summary\n")
        f.write("=======================\n\n")
        f.write("Top features used:\n")
        for feat in feature_names:
            f.write(f" - {feat}\n")
        f.write("\nLogistic Regression ROC-AUC: {:.4f}\n".format(auc_lr))
        f.write("Random Forest ROC-AUC: {:.4f}\n".format(auc_rf))

    print(f"✅ Saved model summary to {summary_path}")


if __name__ == "__main__":
    run_baseline_models()
