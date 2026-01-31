# src/clean_preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_and_preprocess(df):
    print("🔹 Handling missing values...")
    missing_threshold = 0.8
    df = df.loc[:, df.isnull().mean() < missing_threshold]

    print("🔹 Filling missing numeric and categorical values...")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())

    print("🔹 Encoding categorical variables...")
    label_enc = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = label_enc.fit_transform(df[col].astype(str))

    print("🔹 Scaling numeric features...")
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(" Data cleaned and scaled.")
    return df

if __name__ == "__main__":
    df = pd.read_csv("../data/processed/train_merged.csv")
    df_clean = clean_and_preprocess(df)
    df_clean.to_csv("../data/processed/train_clean.csv", index=False)
    print("✅ Saved clean dataset to /data/processed/train_clean.csv")
