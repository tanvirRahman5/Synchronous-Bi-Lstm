# data/preprocess_data_bilstm.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
import os

# -----------------------------
# CONFIG
# -----------------------------
RAW_PATH = "data/raw/crop_fertilizer.csv"
PROCESSED_PATH = "data/processed/cleaned.csv"
PARTITION_DIR = "data/partitions"
NUM_CLIENTS = 4
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
os.makedirs(PARTITION_DIR, exist_ok=True)

# -----------------------------
# 1. Load raw data
# -----------------------------
df = pd.read_csv(RAW_PATH)

# -----------------------------
# 2. Drop unused columns
# -----------------------------
df = df.drop(columns=["Fertilizer", "Link"], errors='ignore')

# -----------------------------
# 3. Separate district
# -----------------------------
districts = df["District_Name"]
df = df.drop(columns=["District_Name"])

# -----------------------------
# 4. Encode target (Crop)
# -----------------------------
label_encoder = LabelEncoder()
df["Crop"] = label_encoder.fit_transform(df["Crop"])

print("Global Crop Label Mapping:")
for crop, label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"{crop} -> {label}")

# -----------------------------
# 5. One-hot encode Soil_color
# -----------------------------
df = pd.get_dummies(df, columns=["Soil_color"], prefix="Soil").astype(int)

# -----------------------------
# 6. Standardize numeric features
# -----------------------------
numeric_cols = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Rainfall", "Temperature"]
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -----------------------------
# 7. Save cleaned dataset (optional)
# -----------------------------
cleaned_df = df.copy()
cleaned_df["District_Name"] = districts.values
cleaned_df.to_csv(PROCESSED_PATH, index=False)
print(f"\nSaved cleaned dataset to {PROCESSED_PATH}")

# -----------------------------
# 8. District-based Non-IID split
# -----------------------------
unique_districts = shuffle(districts.unique(), random_state=RANDOM_SEED)
district_splits = np.array_split(unique_districts, NUM_CLIENTS)

client_data = []
for i in range(NUM_CLIENTS):
    client_df = cleaned_df[cleaned_df["District_Name"].isin(district_splits[i])]
    client_data.append(client_df)

# -----------------------------
# 9. Equalize sample size & enforce column consistency
# -----------------------------
min_samples = min(len(cdf) for cdf in client_data)
print(f"\nEqualizing each client to {min_samples} samples")

all_soil_cols = [col for col in df.columns if col.startswith("Soil_")]

client_inputs = []
client_targets = []

for i in range(NUM_CLIENTS):
    client_df = client_data[i].sample(n=min_samples, random_state=RANDOM_SEED).drop(columns=["District_Name"])

    # Add missing soil columns
    for col in all_soil_cols:
        if col not in client_df.columns:
            client_df[col] = 0

    # Reorder columns: numeric -> soil -> Crop
    client_df = client_df[numeric_cols + all_soil_cols + ["Crop"]]

    # Separate input and target
    X = client_df.drop(columns=["Crop"]).values  # shape: [samples, features]
    y = client_df["Crop"].values                 # shape: [samples]

    # Reshape X to 3D for Bi-LSTM: [samples, timesteps, features_per_timestep]
    # Here, we treat each feature as a timestep with 1 feature
    X = X[:, :, np.newaxis]  # shape -> [samples, features, 1]

    client_inputs.append(X)
    client_targets.append(y)

    # Save as .npz for each client
    np.savez(f"{PARTITION_DIR}/client_{i}_data.npz", X=X, y=y)
    print(f"Client {i}: X shape = {X.shape}, y shape = {y.shape}, Unique crops = {len(np.unique(y))}")

print("\n✅ Data preprocessing & Bi-LSTM-ready partitioning completed successfully.")
