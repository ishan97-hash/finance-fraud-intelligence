# src/load_merge_ieee.py
import pandas as pd

def load_and_merge_ieee(transaction_path, identity_path):
    print("🔹 Loading data...")
    train_transaction = pd.read_csv(transaction_path)
    train_identity = pd.read_csv(identity_path)

    print("🔹 Merging transaction and identity data on 'TransactionID'...")
    train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

    print(f"Merged data shape: {train.shape}")
    return train

if __name__ == "__main__":
    transaction_path = "../data/raw/ieee/train_transaction.csv"
    identity_path = "../data/raw/ieee/train_identity.csv"
    df = load_and_merge_ieee(transaction_path, identity_path)

    df.to_csv("../data/processed/train_merged.csv", index=False)
    print("Saved merged dataset to /data/processed/train_merged.csv")
