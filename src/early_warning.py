import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -----------------------------
# 1. Load cleaned dataset
# -----------------------------
data_path = r"S:\Study\INST_737_Data Science\Finance_Fraud_Project\data\processed\train_clean.csv"
print(f"🔹 Loading cleaned data from {data_path}...")
df = pd.read_csv(data_path)
print("✅ Loaded:", df.shape)

# isFraud is scaled in this file → reconstruct binary label
df["isFraud_binary"] = (df["isFraud"] > 0).astype(int)

# -----------------------------
# 2. Select top features (same as clustering)
# -----------------------------
top_20 = [
    "V257","V246","V244","V242","V201","V200","V45","V189","V86","V87",
    "V258","V188","V44","V228","V170","V52","V171","V199","V51","V230"
]

# sample for efficiency but keep index to join back
print("🔹 Sampling 100000 rows for risk scoring...")
sample = df.sample(100000, random_state=42)
X = sample[top_20].copy()

# -----------------------------
# 3. PCA + KMeans clustering
# -----------------------------
print("🔹 Running PCA (2 components) for clustering...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

print("🔹 Running KMeans (k=3) on PCA space...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X_pca)

sample["cluster"] = clusters

# -----------------------------
# 4. Identify fraud-heavy cluster
# -----------------------------
cluster_stats = sample.groupby("cluster")["isFraud_binary"].agg(["mean", "count"])
print("\n🔹 Fraud rate per cluster:")
print(cluster_stats)

fraud_cluster = cluster_stats["mean"].idxmax()
print(f"\n⚠️ Fraud-heavy cluster identified as: {fraud_cluster}")

fraud_center = kmeans.cluster_centers_[fraud_cluster]

# -----------------------------
# 5. Distance-based risk score
# -----------------------------
# distance to fraud cluster center
distances = np.linalg.norm(X_pca - fraud_center, axis=1)

# normalize distances to [0,1]
dist_norm = (distances - distances.min()) / (distances.max() - distances.min())

# invert: closer to fraud center → higher risk
risk_score = 1.0 - dist_norm

sample["risk_score"] = risk_score

# -----------------------------
# 6. Define risk categories using quantiles
# -----------------------------
q_low  = np.quantile(risk_score, 0.85)  # 85th percentile
q_med  = np.quantile(risk_score, 0.95)  # 95th percentile
q_high = np.quantile(risk_score, 0.99)  # 99th percentile

print("\n🔹 Risk score thresholds:")
print(f"   Low-risk threshold (85%):  {q_low:.4f}")
print(f"   Medium-risk threshold (95%): {q_med:.4f}")
print(f"   High-risk threshold (99%):   {q_high:.4f}")

def assign_risk(score):
    if score >= q_high:
        return "High"
    elif score >= q_med:
        return "Medium"
    elif score >= q_low:
        return "Low"
    else:
        return "Safe"

sample["risk_level"] = sample["risk_score"].apply(assign_risk)

# -----------------------------
# 7. Summary metrics per tier
# -----------------------------
tier_stats = sample.groupby("risk_level")["isFraud_binary"].agg(
    fraud_rate="mean",
    count="count"
).sort_index()

print("\n🔹 Fraud rate per risk tier:")
print(tier_stats)

# -----------------------------
# 8. Plot histogram with thresholds
# -----------------------------
os.makedirs("../outputs/figures", exist_ok=True)

plt.figure(figsize=(6,4))
plt.hist(risk_score, bins=50)
plt.axvline(q_low,  color="green", linestyle="--", label="85% (Low)")
plt.axvline(q_med,  color="orange", linestyle="--", label="95% (Medium)")
plt.axvline(q_high, color="red", linestyle="--", label="99% (High)")
plt.title("Early Warning Risk Score Distribution")
plt.xlabel("Risk score (0 = safe, 1 = highly fraud-like)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/risk_score_dist.png")
plt.close()

# -----------------------------
# 9. Save scored sample to CSV
# -----------------------------
os.makedirs("../outputs", exist_ok=True)
out_cols = ["TransactionID", "isFraud_binary", "risk_score", "risk_level", "cluster"]
output_path = "../outputs/early_warning_scores.csv"
sample[out_cols].to_csv(output_path, index=False)

print(f"\n✅ Early warning risk scores generated and saved to {output_path}")
print("   Histogram saved to ../outputs/figures/risk_score_dist.png")
