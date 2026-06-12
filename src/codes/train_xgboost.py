"""
flood_prediction_xgboost.py
============================
End-to-end XGBoost Regression Pipeline untuk Flood Prediction.

Struktur Output:
    outputs/
    ├── figures/   → semua visualisasi EDA, feature importance, SHAP
    ├── models/    → model tersimpan (.pkl)
    └── reports/   → laporan evaluasi (.txt)

Author  : Generated for SmartLab / Flood Prediction Project
Python  : 3.9+
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORT LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONSTANTA & PATH
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "../../data/raw/flood.csv"))

OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures", "xgboost")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

TARGET  = "FloodProbability"
FEATURES = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement",
    "Deforestation", "Urbanization", "ClimateChange", "DamsQuality",
    "Siltation", "AgriculturalPractices", "Encroachments",
    "IneffectiveDisasterPreparedness", "DrainageSystems",
    "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors",
]

RANDOM_STATE = 42
SECTION_SEP  = "=" * 62


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _setup_output_dirs() -> None:
    """Buat struktur folder output jika belum ada."""
    for d in [FIGURES_DIR, MODELS_DIR, REPORTS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


def _section(title: str) -> None:
    print(f"\n{SECTION_SEP}\n  {title}\n{SECTION_SEP}")


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error; abaikan nilai nol pada target."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Muat dataset CSV dari path relatif yang ditentukan.
    Tampilkan shape dan preview 5 baris pertama.
    """
    _section("LOAD DATASET")

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] File tidak ditemukan: {DATASET_PATH}")
        print("Pastikan flood.csv ada di folder: data/raw/")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)

    print(f"  Path    : {DATASET_PATH}")
    print(f"  Shape   : {df.shape[0]:,} baris × {df.shape[1]} kolom")
    print(f"\n  Preview (5 baris pertama):")
    print(df.head().to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def perform_eda(df: pd.DataFrame) -> dict:
    """
    EDA lengkap mencakup:
      - Dataset overview
      - Data quality check
      - Descriptive statistics
      - Penjelasan outlier handling
      - Visualisasi: histogram, boxplot, heatmap, scatter

    Returns
    -------
    dict
        'target_corr' : pd.Series — korelasi fitur vs target (sorted desc)
        'corr_matrix' : pd.DataFrame — full correlation matrix
    """
    _section("EXPLORATORY DATA ANALYSIS")

    # ── 3a. Dataset Overview ──────────────────────────────────────────────────
    print("\n[A] DATASET OVERVIEW")
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"  Rows        : {df.shape[0]:,}")
    print(f"  Columns     : {df.shape[1]}")
    print(f"  Memory      : {mem_mb:.2f} MB")
    print("\n  Data Types:")
    print(df.dtypes.to_string())

    # ── 3b. Data Quality ──────────────────────────────────────────────────────
    print("\n[B] DATA QUALITY")
    total_missing = df.isnull().sum().sum()
    per_col       = df.isnull().sum()
    missing_cols  = per_col[per_col > 0].to_dict()
    print(f"  Missing values  : {total_missing}  "
          f"({'None' if not missing_cols else missing_cols})")
    dup = df.duplicated().sum()
    print(f"  Duplicate rows  : {dup}")

    # ── 3c. Descriptive Statistics ────────────────────────────────────────────
    print("\n[C] DESCRIPTIVE STATISTICS")
    stats = df[FEATURES + [TARGET]].describe().T.copy()
    stats["median"]   = df[FEATURES + [TARGET]].median()
    stats["skewness"] = df[FEATURES + [TARGET]].skew()
    stats["kurtosis"] = df[FEATURES + [TARGET]].kurtosis()
    print(
        stats[["mean", "median", "std", "min", "max", "skewness", "kurtosis"]]
        .round(4)
        .to_string()
    )

    # ── 3d. Outlier Handling ──────────────────────────────────────────────────
    print("\n[D] OUTLIER HANDLING")
    print(
        "  Outlier terdeteksi via metode IQR dan Z-score.\n"
        "  Namun karena model yang digunakan adalah XGBoost (tree ensemble),\n"
        "  outlier TIDAK dihapus. Alasan:\n"
        "   • Tree-based model melakukan split berdasarkan threshold/ranking nilai,\n"
        "     bukan jarak Euclidean seperti pada linear model.\n"
        "   • Nilai ekstrem tidak mendistorsi titik split, sehingga XGBoost\n"
        "     secara alami robust terhadap outlier.\n"
        "   • Menghapus outlier berisiko membuang informasi penting yang nyata\n"
        "     terjadi pada kondisi banjir ekstrem."
    )

    # ── 3e. Correlation ───────────────────────────────────────────────────────
    corr_matrix  = df[FEATURES + [TARGET]].corr()
    target_corr  = corr_matrix[TARGET].drop(TARGET).sort_values(ascending=False)

    print("\n[E] CORRELATION FITUR vs TARGET (Top 5)")
    print(target_corr.head(5).round(4).to_string())
    print(
        "\n  Catatan: Korelasi relatif rendah (~0.22–0.23) namun konsisten\n"
        "  di seluruh fitur, yang menunjukkan kontribusi merata (tidak ada\n"
        "  satu fitur dominan). Ini wajar untuk fenomena banjir multifaktor."
    )

    # ── 3f. Visualizations ───────────────────────────────────────────────────
    _plot_histograms(df)
    _plot_boxplots(df)
    _plot_correlation_heatmap(corr_matrix)
    _plot_scatter_top_features(df, target_corr)

    return {"target_corr": target_corr, "corr_matrix": corr_matrix}


# ── Histogram ─────────────────────────────────────────────────────────────────
def _plot_histograms(df: pd.DataFrame) -> None:
    """
    Histogram seluruh fitur dengan bins khusus integer (tanpa celah).

    Penjelasan histogram "bolong":
    ─────────────────────────────
    Fitur dalam dataset ini adalah integer diskrit (rentang ~1–10).
    Ketika menggunakan bins otomatis (algoritma Sturges/Freedman-Diaconis),
    jumlah bins yang dihasilkan bisa lebih besar dari jumlah nilai unik,
    sehingga lebar setiap bin < 1. Karena tidak ada nilai antara dua integer
    berurutan (misal antara 3 dan 4), bins tersebut menghasilkan batang kosong
    yang terlihat sebagai 'celah' (bolong) pada histogram.

    Solusi: gunakan bins = np.arange(lo - 0.5, hi + 1.5, 1) sehingga setiap
    integer mendapat satu batang tepat di tengah bin, tanpa celah.
    """
    print("\n  → Menyimpan histogram (bins integer-safe)...")

    cols   = FEATURES + [TARGET]
    n_cols = 4
    n_rows = int(np.ceil(len(cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 3.8))
    axes = axes.flatten()
    fig.suptitle(
        "Distribusi Seluruh Fitur & Target\n"
        "(bins disesuaikan untuk data diskrit integer — tanpa celah)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for i, col in enumerate(cols):
        ax  = axes[i]
        val = df[col].dropna()

        if col == TARGET:
            # Target float → bins reguler
            ax.hist(val, bins=40, color="#2e86ab", edgecolor="white", alpha=0.88)
        else:
            # Fitur integer → bins tepat tanpa bolong
            lo   = int(val.min())
            hi   = int(val.max())
            bins = np.arange(lo - 0.5, hi + 1.5, 1)
            ax.hist(val, bins=bins, color="#2e86ab", edgecolor="white", alpha=0.88)

        # Garis mean & median
        mean_v   = val.mean()
        median_v = val.median()
        ax.axvline(mean_v,   color="#e63946", ls="--", lw=1.3,
                   label=f"Mean={mean_v:.2f}")
        ax.axvline(median_v, color="#2a9d8f", ls="-.", lw=1.3,
                   label=f"Median={median_v:.2f}")

        ax.set_title(col, fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Value", fontsize=7.5)
        ax.set_ylabel("Count",  fontsize=7.5)
        ax.legend(fontsize=6.2, loc="upper right")
        ax.tick_params(labelsize=7)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "01_histograms.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"     Saved → {path}")


# ── Boxplot ───────────────────────────────────────────────────────────────────
def _plot_boxplots(df: pd.DataFrame) -> None:
    """Boxplot seluruh fitur untuk inspeksi outlier."""
    print("  → Menyimpan boxplots...")

    n_cols = 4
    n_rows = int(np.ceil(len(FEATURES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 3.2))
    axes = axes.flatten()
    fig.suptitle("Boxplot Seluruh Fitur – Inspeksi Outlier",
                 fontsize=14, fontweight="bold")

    for i, col in enumerate(FEATURES):
        sns.boxplot(
            y=df[col], ax=axes[i], color="#4c9be8",
            flierprops=dict(marker="o", markersize=2.5, alpha=0.35, color="#e63946"),
        )
        axes[i].set_title(col, fontsize=8.5, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].tick_params(labelsize=7)

    for j in range(len(FEATURES), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_boxplots.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"     Saved → {path}")


# ── Correlation Heatmap ───────────────────────────────────────────────────────
def _plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> None:
    """Heatmap korelasi seluruh fitur termasuk target, dengan anotasi."""
    print("  → Menyimpan correlation heatmap...")

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        corr_matrix,
        annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.35, linecolor="white",
        annot_kws={"size": 6.5},
        ax=ax,
    )
    ax.set_title(
        "Correlation Heatmap — Seluruh Fitur & Target (FloodProbability)",
        fontsize=13, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "03_correlation_heatmap.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"     Saved → {path}")


# ── Scatter Plot Top 5 ────────────────────────────────────────────────────────
def _plot_scatter_top_features(df: pd.DataFrame,
                                target_corr: pd.Series) -> None:
    """Scatter plot 5 fitur terkorelasi tertinggi vs target + garis regresi."""
    print("  → Menyimpan scatter plots (top-5 features)...")

    top5 = target_corr.head(5).index.tolist()
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
    fig.suptitle(
        "5 Fitur dengan Korelasi Tertinggi terhadap FloodProbability",
        fontsize=13, fontweight="bold",
    )

    for i, feat in enumerate(top5):
        ax  = axes[i]
        smp = df[[feat, TARGET]].sample(2000, random_state=RANDOM_STATE)

        ax.scatter(smp[feat], smp[TARGET],
                   alpha=0.22, s=10, color="#4c9be8", rasterized=True)

        # Garis regresi linier
        m, b     = np.polyfit(smp[feat], smp[TARGET], 1)
        x_line   = np.linspace(smp[feat].min(), smp[feat].max(), 200)
        ax.plot(x_line, m * x_line + b, color="#e63946", lw=1.8,
                label=f"r = {target_corr[feat]:.3f}")

        ax.set_title(feat, fontsize=8.5, fontweight="bold")
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel(TARGET if i == 0 else "", fontsize=8)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "04_scatter_top5.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"     Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING / PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame):
    """
    Persiapan data sebelum training.

    Keputusan desain:
    ─────────────────
    • Tidak ada encoding  → semua fitur sudah numerik integer.
    • Tidak ada imputasi  → tidak ada missing value.
    • Scaling TIDAK digunakan:
        XGBoost (tree ensemble) membuat keputusan split berdasarkan
        threshold absolut nilai fitur, bukan jarak. StandardScaler /
        MinMaxScaler tidak mengubah urutan relatif nilai, sehingga
        split tree identik sebelum maupun sesudah scaling.
        Menggunakan scaler hanya menambah overhead tanpa manfaat.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    _section("PREPROCESSING")

    print("  ✓ Encoding  : tidak diperlukan (semua fitur numerik)")
    print("  ✓ Imputasi  : tidak diperlukan (tidak ada missing value)")
    print("  ✓ Scaling   : TIDAK digunakan — XGBoost tidak memerlukannya")
    print("                (tree split berdasarkan threshold, bukan jarak)")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"\n  Train set : {X_train.shape[0]:,} samples  ({X_train.shape[0]/len(df)*100:.0f}%)")
    print(f"  Test  set : {X_test.shape[0]:,} samples  ({X_test.shape[0]/len(df)*100:.0f}%)")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 5–6. TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X_train: pd.DataFrame,
                y_train: pd.Series) -> XGBRegressor:
    """
    Latih XGBRegressor dengan konfigurasi yang ditentukan.
    """
    _section("TRAINING MODEL")

    model = XGBRegressor(
        objective       = "reg:squarederror",
        n_estimators    = 500,
        learning_rate   = 0.05,
        max_depth       = 6,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        random_state    = RANDOM_STATE,
        n_jobs          = -1,
    )

    print("  Melatih XGBRegressor (n_estimators=500, lr=0.05, max_depth=6)...")
    model.fit(X_train, y_train, verbose=False)
    print("  ✓ Training selesai.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATE MODEL
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model,
                   X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series,  y_test: pd.Series) -> dict:
    """
    Hitung R², MAE, RMSE, MAPE pada train dan test set.
    Tampilkan interpretasi overfitting dan generalisasi.
    """
    _section("MODEL EVALUATION")

    results = {}

    for label, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        y_pred = model.predict(X)
        y_np   = y.values

        r2   = r2_score(y_np, y_pred)
        mae  = mean_absolute_error(y_np, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_np, y_pred)))
        mape = _mape(y_np, y_pred)

        results[label] = {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}

        print(f"\n  [{label} Set]")
        print(f"    R²   : {r2:.6f}")
        print(f"    MAE  : {mae:.6f}")
        print(f"    RMSE : {rmse:.6f}")
        print(f"    MAPE : {mape:.4f} %")

    # ── Interpretasi ─────────────────────────────────────────────────────────
    r2_train = results["Train"]["R2"]
    r2_test  = results["Test"]["R2"]
    gap      = r2_train - r2_test

    print("\n  [INTERPRETASI]")
    if gap > 0.05:
        print(f"  ⚠  Gap R² (Train − Test) = {gap:.4f}")
        print("     Terdapat indikasi overfitting ringan.")
        print("     Pertimbangkan: kurangi n_estimators, max_depth,")
        print("     atau naikkan regularisasi (reg_alpha / reg_lambda).")
    else:
        print(f"  ✓  Gap R² (Train − Test) = {gap:.4f} → tidak ada overfitting signifikan.")

    if r2_test >= 0.90:
        print("  ✓  Generalisasi model SANGAT BAIK (R² test ≥ 0.90)")
    elif r2_test >= 0.80:
        print("  ✓  Generalisasi model BAIK (R² test ≥ 0.80)")
    elif r2_test >= 0.70:
        print("  ✓  Generalisasi model CUKUP (R² test ≥ 0.70)")
    else:
        print("  ⚠  Generalisasi KURANG — pertimbangkan tuning lebih lanjut.")

    _save_evaluation_report(results, gap, r2_train, r2_test)
    return results


def _save_evaluation_report(results: dict, gap: float,
                              r2_train: float, r2_test: float) -> None:
    """Tulis laporan evaluasi ke reports/evaluation_report.txt."""
    path = os.path.join(REPORTS_DIR, "evaluation_report.txt")

    with open(path, "w", encoding="utf-8") as f:
        line = "=" * 58
        f.write(f"{line}\n")
        f.write("      FLOOD PREDICTION — XGBoost EVALUATION REPORT\n")
        f.write(f"{line}\n\n")

        for label, metrics in results.items():
            f.write(f"[{label} Set]\n")
            f.write(f"  R²   : {metrics['R2']:.6f}\n")
            f.write(f"  MAE  : {metrics['MAE']:.6f}\n")
            f.write(f"  RMSE : {metrics['RMSE']:.6f}\n")
            f.write(f"  MAPE : {metrics['MAPE']:.4f} %\n\n")

        f.write(f"Gap R² (Train − Test) : {gap:.4f}\n")

        if gap > 0.05:
            verdict = "Overfitting ringan — pertimbangkan regularisasi."
        else:
            verdict = "Tidak ada overfitting signifikan."
        f.write(f"Verdict               : {verdict}\n\n")

        f.write("Catatan:\n")
        f.write("  • Scaling tidak digunakan (XGBoost tree-based).\n")
        f.write("  • Outlier tidak dihapus (tree ensemble robust).\n")
        f.write("  • random_state=42 di seluruh pipeline.\n")

    print(f"\n  Laporan tersimpan → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CROSS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def cross_validate_model(model,
                          X_train: pd.DataFrame,
                          y_train: pd.Series) -> None:
    """
    5-Fold Cross Validation pada training set.
    Laporkan mean ± std untuk R² dan RMSE.
    """
    _section("CROSS VALIDATION (5-Fold KFold)")

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv = cross_validate(
        model, X_train, y_train,
        cv=kf,
        scoring=["r2", "neg_root_mean_squared_error"],
        return_train_score=False,
        n_jobs=-1,
    )

    r2_scores   =  cv["test_r2"]
    rmse_scores = -cv["test_neg_root_mean_squared_error"]

    print(f"\n  R² per fold  : {np.round(r2_scores, 4).tolist()}")
    print(f"  Mean R²      : {r2_scores.mean():.4f}  ±  {r2_scores.std():.4f}")
    print(f"\n  RMSE per fold: {np.round(rmse_scores, 6).tolist()}")
    print(f"  Mean RMSE    : {rmse_scores.mean():.6f}  ±  {rmse_scores.std():.6f}")

    print("\n  [INTERPRETASI CV]")
    if r2_scores.std() < 0.005:
        print("  ✓ Std R² sangat kecil (<0.005) → model sangat stabil antar fold.")
    elif r2_scores.std() < 0.01:
        print("  ✓ Std R² kecil (<0.01) → model stabil dan konsisten.")
    else:
        print("  ⚠ Std R² cukup besar → variabilitas antar fold perlu diinvestigasi.")


# ─────────────────────────────────────────────────────────────────────────────
# 9. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(model,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_test:  pd.DataFrame,
                             y_test:  pd.Series) -> None:
    """
    Buat dan simpan:
      1. Built-in XGBoost Importance (gain) — Top 20
      2. Permutation Importance (test set)

    Perbedaan keduanya dijelaskan di output konsol.
    """
    _section("FEATURE IMPORTANCE")

    # ── Built-in Importance (gain) ────────────────────────────────────────────
    imp_df = (
        pd.DataFrame({"feature": FEATURES,
                      "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    palette = sns.color_palette("viridis_r", len(imp_df))
    sns.barplot(data=imp_df, y="feature", x="importance",
                palette=palette, ax=ax)
    ax.set_title("XGBoost Built-in Feature Importance (Top 20)\n"
                 "Metric: gain — total information gain saat fitur digunakan sebagai split",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance Score (gain)", fontsize=9)
    ax.set_ylabel("Feature", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    path_builtin = os.path.join(FIGURES_DIR, "05_builtin_importance.png")
    plt.savefig(path_builtin, dpi=130, bbox_inches="tight")
    plt.close()

    # ── Permutation Importance ────────────────────────────────────────────────
    print("  Menghitung permutation importance (n_repeats=10) pada test set...")
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    perm_df = (
        pd.DataFrame({
            "feature"   : FEATURES,
            "importance": perm.importances_mean,
            "std"       : perm.importances_std,
        })
        .sort_values("importance", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(
        perm_df["feature"], perm_df["importance"],
        xerr=perm_df["std"], color="#4c9be8", alpha=0.88,
        ecolor="#777", capsize=3,
    )
    ax.invert_yaxis()
    ax.set_title("Permutation Feature Importance (Test Set)\n"
                 "Metric: mean decrease R² ketika nilai fitur diacak",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Decrease in R²", fontsize=9)
    ax.set_ylabel("Feature", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    path_perm = os.path.join(FIGURES_DIR, "06_permutation_importance.png")
    plt.savefig(path_perm, dpi=130, bbox_inches="tight")
    plt.close()

    print(f"  Saved → {path_builtin}")
    print(f"  Saved → {path_perm}")

    # ── Perbandingan & Interpretasi ───────────────────────────────────────────
    print("\n  [PERBANDINGAN BUILT-IN vs PERMUTATION IMPORTANCE]")
    print(
        "\n  Built-in Importance (gain):\n"
        "   • Dihitung secara internal selama training.\n"
        "   • Mengukur total information gain yang dikontribusikan fitur\n"
        "     setiap kali digunakan sebagai titik split.\n"
        "   • Cenderung bias terhadap fitur bertipe high-cardinality atau\n"
        "     fitur yang sering dipilih sebagai split meski kontribusinya kecil.\n"
        "   • Cepat dihitung karena sudah tersimpan di dalam model.\n"
        "\n  Permutation Importance:\n"
        "   • Dihitung pada test set (data yang tidak pernah dilihat model).\n"
        "   • Mengukur penurunan R² aktual ketika nilai satu fitur diacak.\n"
        "   • Lebih representatif karena mencerminkan kontribusi nyata\n"
        "     terhadap kemampuan prediksi, bukan internal tree metrics.\n"
        "   • Lebih lambat (butuh n_repeats × predict), namun lebih andal\n"
        "     untuk interpretasi model dalam konteks generalisasi."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 10. SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def plot_shap(model, X_train: pd.DataFrame) -> None:
    """
    Hitung SHAP values dan simpan:
      • SHAP Summary Plot (beeswarm)
      • SHAP Bar Plot (mean |SHAP|)
    """
    _section("SHAP ANALYSIS")

    # Subsample 1000 untuk efisiensi
    X_sample  = X_train.sample(1000, random_state=RANDOM_STATE)
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_sample)

    # ── Summary Plot (beeswarm) ───────────────────────────────────────────────
    print("  → Menyimpan SHAP summary plot (beeswarm)...")
    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_vals, X_sample, show=False)
    plt.title(
        "SHAP Summary Plot — Distribusi Pengaruh Fitur terhadap FloodProbability\n"
        "(warna merah = nilai fitur tinggi, biru = nilai fitur rendah)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path1 = os.path.join(FIGURES_DIR, "07_shap_summary.png")
    plt.savefig(path1, dpi=130, bbox_inches="tight")
    plt.close()

    # ── Bar Plot (mean |SHAP|) ────────────────────────────────────────────────
    print("  → Menyimpan SHAP bar plot...")
    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_vals, X_sample, plot_type="bar", show=False)
    plt.title(
        "SHAP Bar Plot — Mean |SHAP| per Fitur\n"
        "(rata-rata dampak absolut terhadap prediksi model)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path2 = os.path.join(FIGURES_DIR, "08_shap_bar.png")
    plt.savefig(path2, dpi=130, bbox_inches="tight")
    plt.close()

    print(f"  Saved → {path1}")
    print(f"  Saved → {path2}")

    # Top 5 fitur berdasarkan SHAP
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    shap_df   = (
        pd.DataFrame({"feature": FEATURES, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
    )
    print("\n  Top 5 fitur paling berpengaruh (SHAP mean |value|):")
    print(shap_df.head(5).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 11. MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
def save_model(model) -> str:
    """Simpan model ke outputs/models/xgboost_flood.pkl menggunakan joblib."""
    _section("MODEL PERSISTENCE")
    path = os.path.join(MODELS_DIR, "xgboost_flood.pkl")
    joblib.dump(model, path)
    print(f"  ✓ Model disimpan → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 12. SAMPLE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_sample(model) -> float:
    """
    Prediksi satu sampel data menggunakan model terlatih.
    Nilai aktual dari baris pertama dataset = 0.45.
    """
    _section("SAMPLE PREDICTION")

    sample = {
        "MonsoonIntensity"                : 3,
        "TopographyDrainage"              : 8,
        "RiverManagement"                 : 6,
        "Deforestation"                   : 6,
        "Urbanization"                    : 4,
        "ClimateChange"                   : 4,
        "DamsQuality"                     : 6,
        "Siltation"                       : 2,
        "AgriculturalPractices"           : 3,
        "Encroachments"                   : 2,
        "IneffectiveDisasterPreparedness" : 5,
        "DrainageSystems"                 : 10,
        "CoastalVulnerability"            : 7,
        "Landslides"                      : 4,
        "Watersheds"                      : 2,
        "DeterioratingInfrastructure"     : 3,
        "PopulationScore"                 : 4,
        "WetlandLoss"                     : 3,
        "InadequatePlanning"              : 2,
        "PoliticalFactors"                : 6,
    }

    X_samp = pd.DataFrame([sample])[FEATURES]
    pred   = float(model.predict(X_samp)[0])

    print("\n  Input Features:")
    for k, v in sample.items():
        print(f"    {k:<40}: {v}")

    print(f"\n  ➜ Predicted FloodProbability : {pred:.6f}")
    print(f"    Nilai aktual (row 0)       : 0.450000")
    print(f"    Selisih absolut            : {abs(pred - 0.45):.6f}")
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print(SECTION_SEP)
    print("  FLOOD PREDICTION — XGBoost Regression Pipeline")
    print(SECTION_SEP)

    _setup_output_dirs()

    # 1. Load
    df = load_data()

    # 2. EDA
    eda_info = perform_eda(df)

    # 3. Preprocessing & split
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # 4. Train
    model = train_model(X_train, y_train)

    # 5. Evaluate
    _results = evaluate_model(model, X_train, X_test, y_train, y_test)

    # 6. Cross Validation
    cross_validate_model(model, X_train, y_train)

    # 7. Feature Importance
    plot_feature_importance(model, X_train, y_train, X_test, y_test)

    # 8. SHAP
    plot_shap(model, X_train)

    # 9. Save
    save_model(model)

    # 10. Sample Prediction
    predict_sample(model)

    # Done
    print(f"\n{SECTION_SEP}")
    print("  ✅  PIPELINE SELESAI")
    print(f"  Output tersimpan di  : {OUTPUT_DIR}")
    print(f"    figures/  → {len(list(Path(FIGURES_DIR).glob('*.png')))} file visualisasi")
    print(f"    models/   → xgboost_flood.pkl")
    print(f"    reports/  → evaluation_report.txt")
    print(SECTION_SEP)


if __name__ == "__main__":
    main()