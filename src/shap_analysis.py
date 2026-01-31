import os
import json

import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

print("🔹 Starting SHAP analysis (TreeExplainer, final version)...")

# ----------------------------------------
# 1. Load model
# ----------------------------------------
model_path = "../models/random_forest.pkl"
if not os.path.exists(model_path):
    print(f"❌ Model not found at: {model_path}")
    raise SystemExit

print("🔹 Loading Random Forest model...")
rf = joblib.load(model_path)

# ----------------------------------------
# 2. Load cleaned dataset
# ----------------------------------------
data_path = "../data/processed/train_clean.csv"
if not os.path.exists(data_path):
    print(f"❌ Cleaned dataset NOT found: {data_path}")
    raise SystemExit

print("🔹 Loading cleaned dataset...")
df = pd.read_csv(data_path)

# ----------------------------------------
# 3. Load EXACT feature list used during training
# ----------------------------------------
feature_list_path = "../models/feature_list.json"
if not os.path.exists(feature_list_path):
    print(f"❌ Feature list NOT found at: {feature_list_path}")
    print("   Make sure you re-ran baseline_models.py after adding the JSON save.")
    raise SystemExit

with open(feature_list_path, "r") as f:
    feature_names = json.load(f)

print(f"🔹 Loaded {len(feature_names)} training features for SHAP.")
print("   Features:", feature_names)

X = df[feature_names]

# ----------------------------------------
# 4. Prepare sample for SHAP
# ----------------------------------------
print("🔹 Sampling 3000 rows for SHAP...")
X_sample = X.sample(3000, random_state=42)

# ----------------------------------------
# 5. Run SHAP TreeExplainer
# ----------------------------------------
print("🔹 Initializing TreeExplainer...")
explainer = shap.TreeExplainer(rf)

print("🔹 Computing SHAP values (this may take ~10–20 seconds)...")
values = explainer.shap_values(X_sample, check_additivity=False)

# values can be:
# - list [class0, class1]
# - array (n_samples, n_features)
# - array (n_samples, n_features, n_classes)
if isinstance(values, list):
    shap_mat = values[1]  # class 1 = fraud
else:
    arr = values
    if arr.ndim == 3:
        # shape (n_samples, n_features, n_classes) -> take fraud class
        shap_mat = arr[:, :, 1]
    else:
        shap_mat = arr

print("🔹 SHAP matrix shape:", shap_mat.shape)
print("🔹 X_sample shape:", X_sample.shape)

# ----------------------------------------
# 6. Save SHAP summary plot
# ----------------------------------------
output_dir = "../outputs/figures/"
os.makedirs(output_dir, exist_ok=True)

print("🔹 Saving SHAP summary plot...")
shap.summary_plot(shap_mat, X_sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=300)
plt.close()

print("✅ SHAP summary saved to outputs/figures/shap_summary.png")
print("🎉 SHAP analysis completed successfully!")
