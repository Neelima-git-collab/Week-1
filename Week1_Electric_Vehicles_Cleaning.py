
# ==========================================================
# Week-1 : Data Understanding & Cleaning
# ==========================================================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
# Make sure your dataset file (electric_vehicles_spec_2025.csv.csv) is in the same directory
df = pd.read_csv("electric_vehicles_spec_2025.csv.csv")
print("✅ Dataset Loaded Successfully!")

# Step 3: Display basic info
print("\n--- Dataset Info ---")
df.info()

# Step 4: Preview first few rows
print("\n--- First 5 Rows ---")
print(df.head())

# Step 5: Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Step 6: Handle missing values
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].mean(), inplace=True)

print("\n✅ Missing values handled successfully!")

# Step 7: Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\nTotal duplicate rows: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("✅ Duplicates removed!")

# Step 8: Summary statistics
print("\n--- Summary Statistics ---")
print(df.describe(include='all'))

# Step 9: Correlation heatmap (for numeric features)
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# Step 10: Visualizations
if 'Range_km' in df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df['Range_km'], kde=True, bins=30)
    plt.title("Distribution of Electric Vehicle Range (km)")
    plt.show()

if 'Brand' in df.columns:
    plt.figure(figsize=(10,5))
    sns.countplot(y='Brand', data=df, order=df['Brand'].value_counts().index)
    plt.title("Count of Electric Vehicles by Brand")
    plt.show()

if {'Price_in_USD', 'Range_km'}.issubset(df.columns):
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='Range_km', y='Price_in_USD', hue='Brand', data=df)
    plt.title("Price vs Range (by Brand)")
    plt.show()

# Step 11: Save cleaned data
df.to_csv("cleaned_electric_vehicles_2025.csv", index=False)
print("\n💾 Cleaned dataset saved as 'cleaned_electric_vehicles_2025.csv'")

# Step 12: Key insights summary
print("\n--- Week-1 Insights ---")
print("1️⃣ Dataset successfully cleaned and explored.")
print("2️⃣ Missing and duplicate data handled.")
print("3️⃣ Key numeric relationships visualized (Price vs Range, etc.).")
print("4️⃣ Correlation heatmap shows relationships among numeric features.")
print("✅ Week-1 task completed successfully!")
