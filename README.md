# 🚀 Beyond Detection — Behavioral Fraud Intelligence Platform

Project: A Multivariate and Behavioral Dynamics Analysis of Financial Fraud
Author: Ishan Bhosekar

Executive Summary

Traditional fraud systems answer:

“Is this transaction fraudulent?”

This project reframes fraud detection as a behavioral risk intelligence problem, answering:

Which customers are becoming risky?

How fraud patterns evolve over time?

What signals drive fraud decisions?

How can risk be flagged before fraud occurs?

This platform integrates supervised modeling, behavioral clustering, temporal drift detection, explainable AI, and early-warning risk scoring to build a multi-layer fraud intelligence system rather than a simple classifier.

# 🎯 Key Capabilities

Fraud prediction using supervised ML

Behavioral clustering to detect hidden fraud groups

Temporal drift monitoring for evolving fraud patterns

Explainable AI using SHAP

Early-warning risk scoring engine

# 📊 Key Results

Dataset: 590,540 transactions, 360 engineered features

Fraud rate: ~3.5% (highly imbalanced real-world scenario)

Random Forest ROC-AUC: 0.744

Identified fraud-dominant behavioral cluster with 95% fraud density

Detected feature drift in high-risk signals (V257, V246, id_17, id_35)

Built tiered early-warning risk thresholds (85 / 95 / 99 percentiles)

# 🧠 System Architecture

The system pipeline includes:

Data preprocessing and feature engineering

Exploratory analysis and imbalance assessment

Supervised modeling with interpretability

Behavioral clustering and anomaly discovery

Temporal drift detection

Early-warning risk scoring

This layered approach enables both reactive detection and proactive risk monitoring.

# 🛠 Tech Stack

Python • Pandas • Scikit-learn • SHAP
SMOTE • PCA • KMeans • Matplotlib • Seaborn

⚙ Requirements / Environment

Python 3.10 – 3.11 recommended

Minimum RAM: 16 GB (large dataset)

OS: Windows / macOS / Linux

Environment Setup
python -m venv .venv

 Windows
.venv\Scripts\activate

macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

Sample requirements.txt
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


If SHAP installation fails:

python -m pip install --upgrade pip

# 📁 Dataset Placement

Place raw IEEE-CIS dataset files in:

data/raw/


Required:

train_transaction.csv
train_identity.csv


Optional:

PaySim dataset


After preprocessing:

data/processed/train_clean.csv

# ▶ Recommended Execution Order

Run scripts sequentially:

1️⃣ clean_preprocess.py
→ Merge, clean, encode, scale data

2️⃣ eda_analysis.py
→ Class distribution + correlation heatmap

3️⃣ view_large_csv.py (optional)
→ Preview large dataset

4️⃣ baseline_models.py
→ Logistic Regression + Random Forest

5️⃣ clustering_analysis.py
→ Behavioral clustering

6️⃣ temporal_analysis.py
→ Fraud drift monitoring

7️⃣ shap_analysis.py
→ Model explainability

8️⃣ early_warning.py
→ Risk scoring engine

# 📜 Script Commands

Run from project root:

python src/<script>.py

Preprocessing
python src/clean_preprocess.py


Output:

data/processed/train_clean.csv

EDA
python src/eda_analysis.py


Outputs:

outputs/figures/class_distribution.png
outputs/figures/correlation_heatmap.png

Baseline Modeling
python src/baseline_models.py


Outputs:

models/random_forest.pkl
models/feature_list.json
outputs/model_summary.txt
confusion matrices + ROC plots

Clustering
python src/clustering_analysis.py


Outputs:

cluster visualization
cluster_stats.csv

Temporal Drift
python src/temporal_analysis.py


Outputs:

drift_<feature>.png

SHAP Explainability
python src/shap_analysis.py


Output:

shap_summary.png

Early Warning Risk Scoring
python src/early_warning.py


Outputs:

risk_score_dist.png
early_warning_scores.csv

# 📊 Outputs Directory
outputs/figures/ → Visualizations
models/ → Saved models
cluster_stats.csv → Cluster analysis
early_warning_scores.csv → Risk tiers

# 🧪 Reproducibility

Fixed random seeds ensure consistent runs

All intermediate artifacts saved

Modular pipeline supports extension

#⚠ Troubleshooting

Common fixes:

Missing processed data
→ Run preprocessing first

Directory errors
→ Ensure outputs/ exists

SHAP shape mismatch
→ Regenerate model + feature list

Memory issues
→ Reduce sampling sizes

# 🔮 Future Enhancements

Causal inference modeling

Real-time deployment API

XGBoost/LSTM forecasting

Automated fraud strategy agent

# 📬 Contact

Author: Ishan Bhosekar
GitHub / LinkedIn: (add your links)

⭐ If this project helped you understand fraud intelligence systems, feel free to star the repo!
