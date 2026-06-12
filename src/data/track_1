# ==================================================
# COMPREHENSIVE EDA UNTUK 1 DATASET
# ==================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==================================================
# KONFIGURASI PATH
# ==================================================
# Menggunakan os.path.dirname(__file__) agar path selalu benar dari mana pun script dijalankan
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "../../data/raw/flood.csv")
DATASET_PATH = os.path.normpath(DATASET_PATH)

# Path untuk menyimpan output gambar
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../../reports/figures/track_1")
OUTPUT_DIR = os.path.normpath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# CEK PATH
# ==================================================
if DATASET_PATH.strip() == "":
    print("Path dataset masih kosong.")
    exit()

if not os.path.exists(DATASET_PATH):
    print("File tidak ditemukan:")
    print(DATASET_PATH)
    exit()

# ==================================================
# LOAD DATASET
# ==================================================
df = pd.read_csv(DATASET_PATH)

print("=" * 80)
print("COMPREHENSIVE EDA REPORT: TRACK 1")
print("Nama File :", os.path.basename(DATASET_PATH))
print("=" * 80)

# ==================================================
# 1. Dataset Overview
# ==================================================
print("\n=== 1. Dataset Overview ===")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"Dataset size in memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

print("\nVariables Classification:")
print(f"- Numerical  : {len(numeric_cols)} cols")
print(f"- Categorical: {len(cat_cols)} cols")

print("\nData Types:")
print(df.dtypes)

# ==================================================
# 2. Data Quality Assessment
# ==================================================
print("\n=== 2. Data Quality Assessment ===")
missing_count = df.isnull().sum()
duplicates = df.duplicated().sum()
print(f"Total Duplicate Rows: {duplicates}")
print("Missing Values per Column:")
if missing_count.sum() > 0:
    print(missing_count[missing_count > 0])
else:
    print("No missing values detected in the entire dataset.")

# ==================================================
# 3. Descriptive Statistics
# ==================================================
print("\n=== 3. Descriptive Statistics ===")
if numeric_cols:
    desc = df[numeric_cols].describe().T
    desc['variance'] = df[numeric_cols].var()
    desc['skewness'] = df[numeric_cols].skew()
    desc['kurtosis'] = df[numeric_cols].kurt()
    desc['range'] = desc['max'] - desc['min']
    desc['IQR'] = desc['75%'] - desc['25%']
    print("\nNumerical Features:")
    print(desc[['count', 'mean', '50%', 'std', 'min', 'max', 'skewness', 'kurtosis']])

if cat_cols:
    print("\nCategorical Features:")
    print(df[cat_cols].describe())

# ==================================================
# 5. Outlier Analysis
# ==================================================
print("\n=== 5. Outlier Analysis (Z-Score & IQR) ===")
outliers_iqr = {}
outliers_z = {}

for col in numeric_cols:
    # IQR Method
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers_iqr[col] = ((df[col] < lower) | (df[col] > upper)).sum()
    
    # Z-score Method
    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
    outliers_z[col] = (z_scores > 3).sum()

outlier_df = pd.DataFrame({
    'IQR Outliers Count': pd.Series(outliers_iqr),
    'Z-Score Outliers Count': pd.Series(outliers_z)
})
print(outlier_df)

# ==================================================
# 7 & 8. Bivariate & Multivariate Analysis
# ==================================================
print("\n=== 7 & 8. Bivariate & Multivariate Analysis ===")
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    
    print("\nHigh Positive Correlations (> 0.5):")
    high_corr_pos = corr.unstack().sort_values(ascending=False).drop_duplicates()
    high_corr_pos = high_corr_pos[(high_corr_pos > 0.5) & (high_corr_pos < 1.0)]
    print(high_corr_pos if not high_corr_pos.empty else "None detected.")
    
    print("\nHigh Negative Correlations (< -0.5):")
    high_corr_neg = corr.unstack().sort_values().drop_duplicates()
    high_corr_neg = high_corr_neg[high_corr_neg < -0.5]
    print(high_corr_neg if not high_corr_neg.empty else "None detected.")

    target_col = 'FloodProbability'
    if target_col in df.columns:
        print(f"\nCorrelations with Target Variable ({target_col}):")
        print(corr[target_col].sort_values(ascending=False).head(10))

# ==================================================
# 10. Visualization (Menyimpan Output ke File)
# ==================================================
print("\n=== 10. Visualization ===")
print(f"Menyimpan semua grafik ke folder:\n{OUTPUT_DIR}")

if numeric_cols:
    num_cols_to_plot = numeric_cols[:21] # Maksimal 21 kolom agar grafik tidak terlalu padat
    
    # 1. Histogram
    df[num_cols_to_plot].hist(figsize=(18, 14), bins=20, edgecolor='black', color='skyblue')
    plt.suptitle("Univariate Analysis: Histograms of Numerical Features", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "1_histograms.png"), dpi=300)
    plt.close()
    print("- Saved: 1_histograms.png")

    # 2. Boxplot (Outliers)
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df[num_cols_to_plot], orient="h", palette="Set2")
    plt.title("Outlier Analysis: Boxplots of Numerical Features", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_boxplots.png"), dpi=300)
    plt.close()
    print("- Saved: 2_boxplots.png")

    # 3. Correlation Heatmap
    plt.figure(figsize=(16, 12))
    corr_matrix = df[num_cols_to_plot].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", annot_kws={"size": 8}, cmap='coolwarm', vmin=-1, vmax=1, center=0, linewidths=0.5)
    plt.title("Multivariate Analysis: Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_correlation_heatmap.png"), dpi=300)
    plt.close()
    print("- Saved: 3_correlation_heatmap.png")

print("\nEDA Selesai. Seluruh hasil analisis teks ada di atas, dan hasil gambar tersimpan di folder reports.")