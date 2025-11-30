"""
Generate unscaled version of the data for proper CatBoost training
This keeps Age, Hours per day, and BPM as real interpretable values
"""

import pandas as pd
import numpy as np

print("="*70)
print("CREATING UNSCALED DATA FOR CATBOOST")
print("="*70)

# Load original data
df = pd.read_csv('Data/mxmh_survey_results.csv')

print(f"\nOriginal data: {df.shape}")

# Drop irrelevant columns
df = df.drop(['Timestamp', 'Primary streaming service', 'Permissions', 
              'Music effects', 'Fav genre'], axis=1, errors='ignore')

# Drop rows with missing targets
df = df.dropna(subset=['Anxiety', 'Depression', 'Insomnia', 'OCD'])

# Remove BPM outlier
df = df[df['BPM'] < 1000]

print(f"After cleaning: {df.shape}")

# === ENCODE CATEGORICAL FEATURES (BUT DON'T SCALE NUMERIC) ===

# 1. Binary features: Yes/No → 1/0
binary_cols = ['While working', 'Instrumentalist', 'Composer', 
               'Exploratory', 'Foreign languages']

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
        print(f"Encoded {col}: Yes→1, No→0")

# 2. Frequency features: Never/Rarely/Sometimes/Very frequently → 0/1/2/3
frequency_cols = [col for col in df.columns if 'Frequency [' in col]

for col in frequency_cols:
    df[col] = df[col].map({
        'Never': 0,
        'Rarely': 1,
        'Sometimes': 2,
        'Very frequently': 3
    })

print(f"Encoded {len(frequency_cols)} genre frequency columns: 0-3 scale")

# 3. Fill any remaining missing values
for col in ['Age', 'Hours per day', 'BPM']:
    if df[col].isna().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Filled {col} missing values with median: {median_val}")

# === REORDER COLUMNS ===
feature_cols = ['Age', 'Hours per day', 'BPM'] + binary_cols + frequency_cols
target_cols = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
df = df[feature_cols + target_cols]

# === SAVE ===
output_path = 'Data/listeningData_unscaled.csv'
df.to_csv(output_path, index=False)

print(f"\n✓ Saved to: {output_path}")
print(f"  Shape: {df.shape}")
