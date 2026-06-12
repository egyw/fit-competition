"""
Track 1 – Flood Prediction
Preprocessing & Feature Selection menggunakan Mutual Information
================================================================
Dataset : flood.csv (50.000 baris, 20 fitur, 1 target)
Task    : Regression → prediksi FloodProbability
Output  : Model + plot hasil MI dan evaluasi

Run:
    python preprocess.py

Output disimpan ke ./flood_outputs/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ── Konfigurasi ───────────────────────────────────────────────────────────────

DATA_PATH  = "data/raw/flood.csv"
OUTPUT_DIR = Path("flood_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL   = "FloodProbability"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Threshold MI — fitur dengan skor di bawah ini akan dibuang
# None  → gunakan semua fitur (rekomendasi untuk dataset ini)
# float → misal 0.025 untuk memilih top-14 fitur
MI_THRESHOLD = None

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)


# ── Helper ────────────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot disimpan → {path}")


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── 1. Load Data ──────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    section("1. Load Data")
    df = pd.read_csv(path)
    print(f"  Shape          : {df.shape}")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    print(f"  Target range   : {df[TARGET_COL].min():.3f} – {df[TARGET_COL].max():.3f}")
    print(f"  Target mean    : {df[TARGET_COL].mean():.3f}")
    return df


# ── 2. Split Fitur & Target ───────────────────────────────────────────────────

def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    print(f"\n  Fitur  : {X.shape[1]} kolom")
    print(f"  Sampel : {X.shape[0]} baris")
    return X, y


# ── 3. Mutual Information ─────────────────────────────────────────────────────

def compute_mutual_information(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    section("2. Mutual Information Feature Selection")

    print("  Menghitung MI scores …")
    mi_scores = mutual_info_regression(X, y, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    print(f"\n  {'Fitur':<40} {'MI Score':>10}")
    print(f"  {'-'*50}")
    for feat, score in mi_series.items():
        bar = "█" * int(score * 1000)
        print(f"  {feat:<40} {score:>10.6f}  {bar}")

    print(f"\n  MI Score — Max : {mi_series.max():.6f}")
    print(f"  MI Score — Min : {mi_series.min():.6f}")
    print(f"  MI Score — Mean: {mi_series.mean():.6f}")
    print(f"  MI Score — Std : {mi_series.std():.6f}")

    # Plot MI scores
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e74c3c" if s >= mi_series.mean() + mi_series.std()
              else "#f39c12" if s >= mi_series.mean()
              else "#3498db"
              for s in mi_series.values]
    ax.barh(mi_series.index[::-1], mi_series.values[::-1], color=colors[::-1])
    ax.axvline(mi_series.mean(), color="navy", linestyle="--",
               linewidth=1.5, label=f"Mean = {mi_series.mean():.4f}")
    ax.axvline(mi_series.mean() + mi_series.std(), color="red", linestyle=":",
               linewidth=1.5, label=f"Mean+Std = {mi_series.mean()+mi_series.std():.4f}")
    ax.set_xlabel("Mutual Information Score")
    ax.set_title("Mutual Information Score per Fitur\n(terhadap FloodProbability)", fontweight="bold")
    ax.legend()
    for i, (feat, score) in enumerate(zip(mi_series.index[::-1], mi_series.values[::-1])):
        ax.text(score + 0.0002, i, f"{score:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "01_mutual_information_scores")

    return mi_series


# ── 4. Feature Selection berdasarkan MI ───────────────────────────────────────

def select_features(X: pd.DataFrame, mi_series: pd.Series, threshold=None) -> pd.DataFrame:
    section("3. Feature Selection")

    if threshold is None:
        selected_features = mi_series.index.tolist()
        print(f"  Threshold : None → semua {len(selected_features)} fitur digunakan")
    else:
        selected_features = mi_series[mi_series >= threshold].index.tolist()
        dropped_features  = mi_series[mi_series < threshold].index.tolist()
        print(f"  Threshold      : {threshold}")
        print(f"  Fitur dipilih  : {len(selected_features)}")
        print(f"  Fitur dibuang  : {len(dropped_features)}")
        if dropped_features:
            print(f"  Dibuang        : {dropped_features}")

    print(f"\n  Fitur terpilih :")
    for i, feat in enumerate(selected_features, 1):
        print(f"    {i:>2}. {feat} (MI={mi_series[feat]:.6f})")

    return X[selected_features]


# ── 5. Train-Test Split ───────────────────────────────────────────────────────

def split_train_test(X_selected: pd.DataFrame, y: pd.Series):
    section("4. Train-Test Split")

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"  Train : {X_train.shape[0]} baris")
    print(f"  Test  : {X_test.shape[0]} baris")
    return X_train, X_test, y_train, y_test


# ── 6. Training Model ─────────────────────────────────────────────────────────

def train_models(X_train, X_test, y_train, y_test):
    section("5. Training & Evaluasi Model")

    models = {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0))
        ]),
        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Lasso(alpha=0.001))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n  ▶ {name}")

        # Cross-validation (5-fold) pada data train
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring="r2", n_jobs=-1
        )
        print(f"    CV R² (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit & predict pada test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        print(f"    Test RMSE      : {rmse:.6f}")
        print(f"    Test MAE       : {mae:.6f}")
        print(f"    Test R²        : {r2:.4f}")

        results[name] = {
            "model":   model,
            "y_pred":  y_pred,
            "cv_r2":   cv_scores.mean(),
            "cv_std":  cv_scores.std(),
            "rmse":    rmse,
            "mae":     mae,
            "r2":      r2,
        }

    return results


# ── 7. Evaluasi Visual ────────────────────────────────────────────────────────

def plot_evaluation(results: dict, y_test: pd.Series) -> None:
    section("6. Visualisasi Evaluasi")

    # ── Bar chart perbandingan metrik ──
    names   = list(results.keys())
    r2_vals = [results[n]["r2"]   for n in names]
    rmse_v  = [results[n]["rmse"] for n in names]
    mae_v   = [results[n]["mae"]  for n in names]
    cv_vals = [results[n]["cv_r2"] for n in names]
    cv_std  = [results[n]["cv_std"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = sns.color_palette("Set2", len(names))

    for ax, vals, title, ylabel in zip(
        axes,
        [r2_vals, rmse_v, mae_v],
        ["Test R² Score", "Test RMSE", "Test MAE"],
        ["R²", "RMSE", "MAE"]
    ):
        bars = ax.bar(names, vals, color=palette, edgecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(names, rotation=20, ha="right")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Perbandingan Performa Model", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "02_model_comparison")

    # ── CV R² dengan error bar ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, cv_vals, yerr=cv_std, color=palette,
           edgecolor="white", capsize=5)
    ax.set_title("Cross-Validation R² (5-Fold) ± Std", fontweight="bold")
    ax.set_ylabel("CV R²")
    ax.set_xticklabels(names, rotation=20, ha="right")
    for i, (v, s) in enumerate(zip(cv_vals, cv_std)):
        ax.text(i, v + s + 0.001, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "03_cv_r2_comparison")

    # ── Actual vs Predicted (setiap model) ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for ax, (name, res) in zip(axes, results.items()):
        y_pred = res["y_pred"]
        ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="#3498db")
        lims = [min(y_test.min(), y_pred.min()) - 0.01,
                max(y_test.max(), y_pred.max()) + 0.01]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual FloodProbability")
        ax.set_ylabel("Predicted FloodProbability")
        ax.set_title(f"{name}\nR²={res['r2']:.4f} | RMSE={res['rmse']:.4f}",
                     fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("Actual vs Predicted — FloodProbability", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "04_actual_vs_predicted")

    # ── Residual plot model terbaik ──
    best_name = max(results, key=lambda n: results[n]["r2"])
    best_pred = results[best_name]["y_pred"]
    residuals = y_test.values - best_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(best_pred, residuals, alpha=0.3, s=10, color="#e74c3c")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residual Plot — {best_name}", fontweight="bold")

    axes[1].hist(residuals, bins=50, color="#9b59b6", edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frekuensi")
    axes[1].set_title(f"Distribusi Residual — {best_name}", fontweight="bold")

    fig.suptitle(f"Residual Analysis — Model Terbaik: {best_name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "05_residual_analysis")


# ── 8. Feature Importance (Random Forest) ────────────────────────────────────

def plot_feature_importance(results: dict, feature_names: list,
                             mi_series: pd.Series) -> None:
    section("7. Feature Importance vs MI Score")

    rf_model = results["Random Forest"]["model"]
    importances = rf_model.feature_importances_
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    # Normalisasi ke 0-1 untuk perbandingan
    mi_norm  = (mi_series[feature_names] - mi_series[feature_names].min()) / \
               (mi_series[feature_names].max() - mi_series[feature_names].min())
    imp_norm = (imp_series - imp_series.min()) / \
               (imp_series.max() - imp_series.min())

    compare_df = pd.DataFrame({
        "Random Forest Importance (norm)": imp_norm,
        "Mutual Information (norm)":       mi_norm[imp_norm.index],
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # RF Importance
    axes[0].barh(imp_series.index[::-1], imp_series.values[::-1], color="#2ecc71")
    axes[0].set_xlabel("Importance Score")
    axes[0].set_title("Random Forest\nFeature Importance", fontweight="bold")
    for i, (feat, val) in enumerate(zip(imp_series.index[::-1], imp_series.values[::-1])):
        axes[0].text(val + 0.0001, i, f"{val:.4f}", va="center", fontsize=8)

    # Perbandingan MI vs RF (grouped bar)
    x = np.arange(len(compare_df))
    width = 0.4
    axes[1].barh(x + width/2, compare_df["Random Forest Importance (norm)"].values,
                 width, label="RF Importance", color="#2ecc71", alpha=0.85)
    axes[1].barh(x - width/2, compare_df["Mutual Information (norm)"].values,
                 width, label="MI Score", color="#3498db", alpha=0.85)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(compare_df.index, fontsize=9)
    axes[1].set_xlabel("Normalized Score (0–1)")
    axes[1].set_title("MI Score vs RF Importance\n(Normalized)", fontweight="bold")
    axes[1].legend()

    fig.suptitle("Feature Importance Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "06_feature_importance_vs_mi")

    print("\n  Perbandingan ranking MI vs RF Importance:")
    mi_rank  = mi_series[feature_names].rank(ascending=False).astype(int)
    imp_rank = imp_series.rank(ascending=False).astype(int)
    rank_df  = pd.DataFrame({
        "MI Rank": mi_rank,
        "RF Rank": imp_rank,
        "Rank Diff": (mi_rank - imp_rank).abs()
    }).sort_values("MI Rank")
    print(rank_df.to_string())


# ── 9. Ringkasan ──────────────────────────────────────────────────────────────

def print_summary(results: dict, mi_series: pd.Series,
                  selected_features: list) -> None:
    section("Ringkasan")

    best_name = max(results, key=lambda n: results[n]["r2"])
    best      = results[best_name]

    print(f"""
  ┌──────────────────────────────────────────────────────┐
  │          TRACK 1 – FLOOD PREDICTION SUMMARY          │
  ├──────────────────────────────────────────────────────┤
  │  Metode Feature Selection : Mutual Information       │
  │  Threshold MI             : {str(MI_THRESHOLD):<26}│
  │  Fitur terpilih           : {len(selected_features):<26}│
  ├──────────────────────────────────────────────────────┤
  │  Model Terbaik            : {best_name:<26}│
  │  Test R²                  : {best['r2']:<26.4f}│
  │  Test RMSE                : {best['rmse']:<26.6f}│
  │  Test MAE                 : {best['mae']:<26.6f}│
  │  CV R² (5-fold)           : {best['cv_r2']:.4f} ± {best['cv_std']:.4f}           │
  ├──────────────────────────────────────────────────────┤
  │  Top-3 Fitur (MI Score)                              │""")

    for feat, score in mi_series.head(3).items():
        short = feat[:35] + "…" if len(feat) > 35 else feat
        print(f"  │    • {short:<37} {score:.4f}  │")

    print(f"  └──────────────────────────────────────────────────────┘")
    print(f"\n  Semua plot disimpan ke: {OUTPUT_DIR.resolve()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Split fitur & target
    X, y = split_features_target(df)

    # 3. Hitung Mutual Information
    mi_series = compute_mutual_information(X, y)

    # 4. Feature selection berdasarkan MI
    X_selected = select_features(X, mi_series, threshold=MI_THRESHOLD)

    # 5. Train-test split
    X_train, X_test, y_train, y_test = split_train_test(X_selected, y)

    # 6. Training & evaluasi semua model
    results = train_models(X_train, X_test, y_train, y_test)

    # 7. Visualisasi evaluasi
    plot_evaluation(results, y_test)

    # 8. Feature importance vs MI
    plot_feature_importance(results, X_selected.columns.tolist(), mi_series)

    # 9. Ringkasan
    print_summary(results, mi_series, X_selected.columns.tolist())


if __name__ == "__main__":
    main()