import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

def load_data(path):
    print(f"🔹 Loading cleaned data from {path}")
    df = pd.read_csv(path)
    print("✅ Loaded:", df.shape)
    return df

def build_X_y_for_clustering(df, top_n=20, sample_n=80000, random_state=42):
    # reconstruct binary label from scaled isFraud
    y_full = (df["isFraud"] > 0).astype(int)

    # pick top correlated features
    corr = df.corr(numeric_only=True)["isFraud"].abs().sort_values(ascending=False)
    feature_names = corr.index[1: top_n + 1].tolist()

    print(f"🔹 Using top {top_n} features for clustering:")
    for f in feature_names:
        print("   -", f)

    X_full = df[feature_names].copy()

    # sample rows for clustering to make KMeans tractable
    if sample_n is not None and sample_n < len(df):
        print(f"🔹 Sampling {sample_n} rows for clustering...")
        sample_idx = df.sample(n=sample_n, random_state=random_state).index
        X = X_full.loc[sample_idx].reset_index(drop=True)
        y = y_full.loc[sample_idx].reset_index(drop=True)
    else:
        print("🔹 Using full dataset for clustering (may be slow)...")
        X = X_full
        y = y_full

    print("   X shape after sampling:", X.shape)
    return X, y, feature_names

def run_pca_kmeans(X, y, n_clusters=3, random_state=42):
    print("🔹 Running PCA -> 2 components for visualization...")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    print(f"🔹 KMeans clustering with k = {n_clusters}")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=5,       # fewer initializations
        max_iter=150    # fewer iterations
    )
    cluster_labels = kmeans.fit_predict(X_pca)

    sil = silhouette_score(X_pca, cluster_labels)
    print("🔹 Silhouette score:", sil)

    # fraud rate per cluster
    df_tmp = pd.DataFrame({
        "cluster": cluster_labels,
        "isFraud": y
    })
    cluster_stats = df_tmp.groupby("cluster")["isFraud"].agg(["mean", "count"])
    print("\n🔹 Fraud rate per cluster (on sample):")
    print(cluster_stats)

    return X_pca, cluster_labels, sil, cluster_stats

def plot_clusters(X_pca, cluster_labels, sil, file_path):
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=cluster_labels,
        alpha=0.6
    )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"KMeans Clusters (Silhouette = {sil:.3f})")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"✅ Saved cluster plot to {file_path}")

if __name__ == "__main__":
    ensure_dirs()
    df = load_data(r"S:\Study\INST_737_Data Science\Finance_Fraud_Project\data\processed\train_clean.csv")

    # sample_n controls how many rows you cluster on
    X, y, feature_names = build_X_y_for_clustering(df, top_n=20, sample_n=80000)

    X_pca, cluster_labels, sil, cluster_stats = run_pca_kmeans(
        X, y, n_clusters=3
    )

    plot_clusters(
        X_pca,
        cluster_labels,
        sil,
        file_path="outputs/figures/pca_kmeans_clusters.png"
    )

    cluster_stats.to_csv("outputs/cluster_stats.csv")
    print("✅ Saved cluster stats to outputs/cluster_stats.csv")
