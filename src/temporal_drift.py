import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"S:\Study\INST_737_Data Science\Finance_Fraud_Project\data\processed\train_clean.csv")

# Convert TransactionDT into "time buckets"
df["time_block"] = pd.qcut(df["TransactionDT"], 10, labels=False)

# Select a few important fraud-related features
key_features = ["V257", "V246", "V201", "V45", "id_35", "id_17"]

# Plot mean drift over time
for feat in key_features:
    plt.figure(figsize=(8,4))
    sns.lineplot(x="time_block", y=feat, hue="isFraud", data=df)
    plt.title(f"Temporal Drift of {feat} Over Time")
    plt.savefig(f"../outputs/figures/drift_{feat}.png")
    plt.close()

print("✅ Drift plots saved.")
