import pandas as pd
import os

# Path to your large CSV
file_path = r"S:\Study\INST_737_Data Science\Finance_Fraud_Project\data\processed\train_clean.csv"


# Number of rows to preview
n_rows = 100

print(f"🔹 Previewing first {n_rows} rows from {file_path}...\n")

# Use chunksize to load large file efficiently
chunk_iter = pd.read_csv(file_path, chunksize=n_rows)

# Read first chunk
df_preview = next(chunk_iter)

# Display basic info
print("✅ File successfully loaded.\n")
print("🔹 Shape:", df_preview.shape)
print("\n🔹 Columns:\n", df_preview.columns.tolist()[:15], "...")  # show only first 15 columns

# Show sample data
print("\n🔹 Sample Data:")
print(df_preview.head())

# Show column data types
print("\n🔹 Data Types:")
print(df_preview.dtypes.head(10))

save_small = input("\n💾 Do you want to save first 500 rows to a smaller CSV for Excel viewing? (y/n): ").strip().lower()
if save_small == 'y':
    small_df = pd.read_csv(file_path, nrows=500)

    # Automatically create folder if missing
    output_dir = os.path.join(os.path.dirname(file_path))
    os.makedirs(output_dir, exist_ok=True)

    small_path = os.path.join(output_dir, "train_clean_sample.csv")
    small_df.to_csv(small_path, index=False)
    print(f"✅ Saved sample file as '{small_path}'")
