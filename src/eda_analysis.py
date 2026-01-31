# src/eda_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    print("🔹 Basic dataset info:")
    print(df.info())
    print(df.describe())

    print("🔹 Checking class balance...")
    sns.countplot(x='isFraud', data=df)
    plt.title("Fraud vs Non-Fraud Distribution")
    plt.savefig("../outputs/figures/class_distribution.png")
    plt.close()

    print("🔹 Correlation heatmap (top 20 features)...")
    corr = df.corr(numeric_only=True)
    top_corr = corr['isFraud'].abs().sort_values(ascending=False).head(20).index
    sns.heatmap(df[top_corr].corr(), cmap='coolwarm', annot=False)
    plt.title("Top Feature Correlations with Fraud")
    plt.savefig("../outputs/figures/correlation_heatmap.png")
    plt.close()

    print(" EDA visuals saved in outputs/figures/")

if __name__ == "__main__":
    df = pd.read_csv("../data/processed/train_clean.csv")
    run_eda(df)
