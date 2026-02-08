Finance Fraud Project — README

Project: Beyond Detection — A Multivariate and Behavioral Dynamics Analysis of Financial Fraud
Author: Ishan Bhosekar

This README explains how to run the analysis code used in the project (EDA, preprocessing, baseline models, clustering, temporal analysis, SHAP, and early-warning scoring) and where to find outputs.

Table of contents

Requirements / Environment

File & folder structure (expected)

Dataset placement

Step-by-step run order (recommended)

Script descriptions & commands

Where outputs are saved

Troubleshooting & tips

Contact

Requirements / Environment

Python 3.10 – 3.11 recommended (works on 3.8+ in most cases).

Minimum RAM: 16 GB recommended (the dataset is large; more memory helps).

OS: Windows, macOS, or Linux.

Create a fresh virtual environment and install dependencies:

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install required libraries
pip install -r requirements.txt


Sample requirements.txt (save in project root):

pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
imbalanced-learn
shap
pyarrow
openpyxl
scipy
notebook
xgboost
lightgbm


Note: shap can be slow to install (compiles some components). If pip install fails, update pip and try again:
python -m pip install --upgrade pip then re-run install.

Dataset placement

Place the original IEEE-CIS CSV files in data/raw/. Typical files:

data/raw/train_transaction.csv

data/raw/train_identity.csv

(optionally) PaySim dataset if used for simulation

Important: The scripts expect train_clean.csv under data/processed/ once preprocessing completes. The preprocessing script will create it.

Recommended run order (high-level)

clean_preprocess.py — merge, clean, encode, scale → produces data/processed/train_clean.csv

eda_analysis.py — basic EDA plots (class distribution, correlation heatmap)

view_large_csv.py — preview large cleaned CSV, optionally save a small sample for Excel

baseline_models.py — build baseline models (LogReg + RandomForest), save RF model and feature list

clustering_analysis.py — PCA + KMeans clustering, save cluster plot and stats

temporal_analysis.py — compute feature drift plots over time blocks

shap_analysis.py — compute and save SHAP summary plot (requires RF model + feature_list.json)

early_warning.py — compute distance-based risk scores, thresholds, save CSV + histogram

Run them sequentially. Later scripts depend on artifacts from earlier ones (e.g., train_clean.csv, random_forest.pkl, feature_list.json).

Script descriptions and commands

All commands assume you run them from the src/ folder or from project root with python src/<script>.py.

1. Preprocessing

Script: src/clean_preprocess.py
Purpose: Merge raw tables, impute missing values, encode categoricals, scale numeric columns, save data/processed/train_clean.csv.
Run:

python src/clean_preprocess.py


Output: data/processed/train_clean.csv and logs about shape and memory usage.

2. EDA

Script: src/eda_analysis.py
Purpose: Produce class distribution chart and correlation heatmap for top features.
Run:

python src/eda_analysis.py


Output: outputs/figures/class_distribution.png, outputs/figures/correlation_heatmap.png.

3. View large CSV (optional)

Script: src/view_large_csv.py
Purpose: Preview first N rows and optionally save a smaller CSV for Excel. Useful if train_clean.csv is too large for Excel/VSCode.
Run:

python src/view_large_csv.py


Interactive: The script may ask if you want to save a sample (y/n). It saves data/processed/train_clean_sample.csv.

4. Baseline models

Script: src/baseline_models.py
Purpose: Select top correlated features, split and apply SMOTE, train Logistic Regression & Random Forest, save model and feature list.
Run:

python src/baseline_models.py


Outputs:

models/random_forest.pkl

models/feature_list.json (list of top 30 features)

outputs/figures/logreg_confusion.png

outputs/figures/logreg_roc.png

outputs/figures/rf_confusion.png

outputs/figures/rf_roc.png

outputs/model_summary.txt

Notes: If the script errors trying to save into a missing directory, ensure outputs/ and models/ directories exist or let the script create them (it should).

5. Clustering analysis

Script: src/clustering_analysis.py
Purpose: PCA, sample large rows for clustering, run KMeans, compute silhouette score, save cluster plot and cluster stats CSV.
Run:

python src/clustering_analysis.py


Outputs:

outputs/figures/pca_kmeans_clusters.png

outputs/cluster_stats.csv

Notes: The script samples (e.g., 80k rows) to speed up clustering; it saves cluster stats for interpretation.

6. Temporal drift analysis

Script: src/temporal_analysis.py
Purpose: Divide TransactionDT into time blocks; compute mean/CI for selected features for fraud vs non-fraud; save per-feature plots.
Run:

python src/temporal_analysis.py


Outputs: outputs/figures/drift_<feature>.png for each analyzed feature (e.g., id_17, id_35, V45, etc.)

7. SHAP interpretability

Script: src/shap_analysis.py
Purpose: Load models/random_forest.pkl and models/feature_list.json, sample rows, compute SHAP TreeExplainer, and save a SHAP summary plot for the fraud class.
Run:

python src/shap_analysis.py


Output: outputs/figures/shap_summary.png

Notes & Troubleshooting:

SHAP expects the same feature order and shape the model was trained on. The script loads feature_list.json to ensure exact ordering.

SHAP computation may be slow and memory heavy for large samples. The script samples 3k rows by default.

If you see "shape mismatch" errors, make sure models/feature_list.json was produced by baseline_models.py and matches the features saved to the model.

8. Early warning scoring

Script: src/early_warning.py
Purpose: Use PCA + KMeans results (fraud cluster) to compute distance-based risk scores, define percentile thresholds (85%, 95%, 99%), assign risk levels, save histogram and CSV.
Run:

python src/early_warning.py


Outputs:

outputs/figures/risk_score_dist.png

outputs/early_warning_scores.csv

Notes: The script samples N rows (100k) for scoring and saves early_warning_scores.csv with columns TransactionID, isFraud_binary, risk_score, risk_level, and cluster.

Where to find outputs

All plots: outputs/figures/ (png images)

Cluster stats: outputs/cluster_stats.csv

Model(s): models/random_forest.pkl, models/feature_list.json

Early warning scores: outputs/early_warning_scores.csv

Small CSV sample (if saved): data/processed/train_clean_sample.csv

Troubleshooting & tips

FileNotFoundError on train_clean.csv: Run clean_preprocess.py first or check that data/processed/train_clean.csv exists and path string matches script expectation.

Permission / path issues on Windows: Use absolute paths in scripts or run PowerShell as Administrator. Prefer python src/<script> from repository root.

Matplotlib savefig errors (no such directory): Create required directories (outputs/figures/) or run ensure_dirs() if script provides it.

SHAP errors (additivity/shape): Ensure models/feature_list.json exists and the model was trained on the same ordered feature list. Re-run baseline_models.py to regenerate model + feature list.

Memory errors: Lower sampling sizes in clustering/SHAP/early_warning scripts or run on a machine with more RAM.

Long runtimes: RF training (n_estimators=200) and SHAP TreeExplainer can take several minutes to hours. Reduce n_estimators or sample fewer rows for SHAP if needed.

Reproducibility notes

Scripts save outputs and models to the project so you can reproduce later analyses.

Keep the same random seeds (many scripts use random_state=42) to get reproducible sampling, splits, and clustering seeds.
