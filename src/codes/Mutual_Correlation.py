"""
Track 1 – Flood Prediction
Preprocessing & Feature Selection menggunakan Mutual Information
+ Evaluasi dengan Repeated K-Fold Cross Validation
================================================================
Dataset      : flood.csv (50.000 baris, 20 fitur, 1 target)
Task         : Regression → prediksi FloodProbability
CV Strategy  : RepeatedKFold(n_splits=10, n_repeats=3) = 30 total fold
Output       : Model + plot hasil MI dan evaluasi

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
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
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

# ── Repeated K-Fold Config ────────────────────────────────────────────────────
N_SPLITS  = 10   # jumlah fold
N_REPEATS = 3    # jumlah pengulangan → total 30 fold
# Total evaluasi = N_SPLITS × N_REPEATS = 30 iterasi

# Threshold MI — fitur dengan skor di bawah ini akan dibuang
# None  → gunakan semua fitur (rekomendasi untuk dataset ini)
# float → misal 0.025 untuk memilih top-14 fitur
MI_THRESHOLD = None

# Train-test split ratio (digunakan untuk final evaluation)
TEST_SIZE = 0.2

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
    section("2. Split Fitur & Target")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    print(f"  Fitur  : {X.shape[1]} kolom")
    print(f"  Sampel : {X.shape[0]} baris")
    return X, y


# ── 3. Train-Test Split ───────────────────────────────────────────────────────

def split_train_test(X: pd.DataFrame, y: pd.Series):
    """
    Membagi data menjadi train dan test set.
    - Train set digunakan untuk Repeated K-Fold CV
    - Test set digunakan untuk evaluasi final (hold-out)
    """
    section("3. Train-Test Split")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"  Strategi       : Hold-out ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})")
    print(f"  Train set      : {X_train.shape[0]:,} baris ({int((1-TEST_SIZE)*100)}%)")
    print(f"  Test set       : {X_test.shape[0]:,} baris ({int(TEST_SIZE*100)}%)")
    print(f"  Random state   : {RANDOM_STATE}")
    print(f"\n  ⚠  Test set HANYA digunakan untuk evaluasi final (hold-out).")
    print(f"     Repeated K-Fold dijalankan di atas train set saja.")

    return X_train, X_test, y_train, y_test


# ── 4. Mutual Information ─────────────────────────────────────────────────────

def compute_mutual_information(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    section("4. Mutual Information Feature Selection")

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


# ── 5. Feature Selection berdasarkan MI ───────────────────────────────────────

def select_features(X: pd.DataFrame, mi_series: pd.Series, threshold=None) -> pd.DataFrame:
    section("5. Feature Selection")

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


# ── 6. Setup Repeated K-Fold ──────────────────────────────────────────────────

def setup_kfold() -> RepeatedKFold:
    section(f"6. Repeated K-Fold Setup")

    rkf = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )
    print(f"  Strategi       : RepeatedKFold")
    print(f"  n_splits       : {N_SPLITS}  (jumlah fold per repeat)")
    print(f"  n_repeats      : {N_REPEATS}  (jumlah pengulangan)")
    print(f"  Total iterasi  : {N_SPLITS * N_REPEATS} fold")
    print(f"  Random state   : {RANDOM_STATE}")
    print(f"\n  Ilustrasi pembagian data per fold (dari train set ~40.000 baris):")
    train_n = int(50000 * (1 - TEST_SIZE))
    print(f"    Train per fold : ~{int(train_n * (N_SPLITS-1)/N_SPLITS):,} baris ({(N_SPLITS-1)/N_SPLITS*100:.0f}%)")
    print(f"    Val per fold   : ~{int(train_n / N_SPLITS):,} baris ({1/N_SPLITS*100:.0f}%)")
    print(f"\n  Penjelasan Repeated K-Fold:")
    print(f"    • Repeat 1  : Fold  1– {N_SPLITS}  (shuffle acak ke-1)")
    print(f"    • Repeat 2  : Fold {N_SPLITS+1}–{N_SPLITS*2}  (shuffle acak ke-2)")
    print(f"    • Repeat 3  : Fold {N_SPLITS*2+1}–{N_SPLITS*3}  (shuffle acak ke-3)")
    print(f"    → Total {N_SPLITS*N_REPEATS} evaluasi independen per model")
    return rkf


# ── 7. Training & Evaluasi dengan Repeated K-Fold ────────────────────────────

def train_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series,
                 rkf: RepeatedKFold) -> dict:
    """
    Melatih dan mengevaluasi semua model menggunakan:
    1. Repeated K-Fold CV pada train set → estimasi performa yang robust
    2. Hold-out test set → evaluasi final tidak bias
    """
    section("7. Training & Evaluasi Model (Repeated K-Fold)")

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

    scoring = {
        "r2":   "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae":  "neg_mean_absolute_error",
    }

    results = {}
    X_train_arr = X_train.values
    X_test_arr  = X_test.values
    y_train_arr = y_train.values
    y_test_arr  = y_test.values

    print(f"  CV Strategy : RepeatedKFold(n_splits={N_SPLITS}, n_repeats={N_REPEATS})")
    print(f"  Total fold  : {N_SPLITS * N_REPEATS} per model\n")

    for name, model in models.items():
        print(f"  ▶ {name}")
        print(f"    Menjalankan {N_SPLITS}×{N_REPEATS} = {N_SPLITS*N_REPEATS} fold CV …",
              end="", flush=True)

        # ── Repeated K-Fold Cross Validation (pada train set) ──
        cv_result = cross_validate(
            model, X_train_arr, y_train_arr,
            cv=rkf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True
        )

        r2_scores   =  cv_result["test_r2"]
        rmse_scores = -cv_result["test_rmse"]
        mae_scores  = -cv_result["test_mae"]
        r2_train    =  cv_result["train_r2"]

        print(" selesai.")

        # ── Hasil CV ──
        print(f"\n    [Repeated K-Fold CV — {N_SPLITS*N_REPEATS} fold]")
        print(f"    {'Metrik':<8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"    {'-'*52}")
        print(f"    {'R²':<8} {r2_scores.mean():>10.6f} {r2_scores.std():>10.6f} {r2_scores.min():>10.6f} {r2_scores.max():>10.6f}")
        print(f"    {'RMSE':<8} {rmse_scores.mean():>10.6f} {rmse_scores.std():>10.6f} {rmse_scores.min():>10.6f} {rmse_scores.max():>10.6f}")
        print(f"    {'MAE':<8} {mae_scores.mean():>10.6f} {mae_scores.std():>10.6f} {mae_scores.min():>10.6f} {mae_scores.max():>10.6f}")
        print(f"    Train R² (mean) : {r2_train.mean():.6f}")
        print(f"    Overfit gap     : {r2_train.mean() - r2_scores.mean():.6f}")

        # ── Evaluasi Hold-out Test Set (final) ──
        model.fit(X_train_arr, y_train_arr)
        y_pred = model.predict(X_test_arr)

        test_r2   = r2_score(y_test_arr, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred))
        test_mae  = mean_absolute_error(y_test_arr, y_pred)

        print(f"\n    [Hold-out Test Set Evaluation]")
        print(f"    Test R²   : {test_r2:.6f}")
        print(f"    Test RMSE : {test_rmse:.6f}")
        print(f"    Test MAE  : {test_mae:.6f}")
        print()

        results[name] = {
            "model":       model,
            # CV metrics (Repeated K-Fold)
            "r2_scores":   r2_scores,
            "rmse_scores": rmse_scores,
            "mae_scores":  mae_scores,
            "r2_train":    r2_train,
            # CV summary
            "cv_r2":       r2_scores.mean(),
            "cv_r2_std":   r2_scores.std(),
            "cv_rmse":     rmse_scores.mean(),
            "cv_mae":      mae_scores.mean(),
            # Hold-out test metrics
            "test_r2":     test_r2,
            "test_rmse":   test_rmse,
            "test_mae":    test_mae,
            # Shorthand (gunakan CV mean sebagai representasi utama)
            "r2":          r2_scores.mean(),
            "r2_std":      r2_scores.std(),
            "rmse":        rmse_scores.mean(),
            "mae":         mae_scores.mean(),
        }

    return results


# ── 8. Evaluasi Visual ────────────────────────────────────────────────────────

def plot_evaluation(results: dict) -> None:
    section("8. Visualisasi Evaluasi")

    names   = list(results.keys())
    palette = sns.color_palette("Set2", len(names))

    # ── 1. Bar chart perbandingan metrik mean CV ──
    r2_vals  = [results[n]["cv_r2"]   for n in names]
    rmse_v   = [results[n]["cv_rmse"] for n in names]
    mae_v    = [results[n]["cv_mae"]  for n in names]
    r2_stds  = [results[n]["cv_r2_std"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, vals, stds, title, ylabel in zip(
        axes,
        [r2_vals, rmse_v, mae_v],
        [r2_stds, [0]*len(names), [0]*len(names)],
        ["Mean R² (30 fold)", "Mean RMSE (30 fold)", "Mean MAE (30 fold)"],
        ["R²", "RMSE", "MAE"]
    ):
        bars = ax.bar(names, vals, color=palette, edgecolor="white",
                      yerr=stds if title.startswith("Mean R²") else None,
                      capsize=5)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(names, rotation=20, ha="right")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle(
        f"Perbandingan Performa Model (Repeated K-Fold CV)\n"
        f"n_splits={N_SPLITS} × n_repeats={N_REPEATS} = {N_SPLITS*N_REPEATS} fold",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    save_fig(fig, "02_model_comparison_cv")

    # ── 2. CV vs Hold-out Test Comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_pairs = [
        ("cv_r2",   "test_r2",   "R² Score",  "R² : CV vs Test"),
        ("cv_rmse", "test_rmse", "RMSE",       "RMSE : CV vs Test"),
        ("cv_mae",  "test_mae",  "MAE",        "MAE : CV vs Test"),
    ]
    x = np.arange(len(names))
    width = 0.35
    for ax, (cv_key, test_key, ylabel, title) in zip(axes, metrics_pairs):
        cv_vals   = [results[n][cv_key]   for n in names]
        test_vals = [results[n][test_key] for n in names]
        ax.bar(x - width/2, cv_vals,   width, label="CV Mean",   color="#3498db", alpha=0.85)
        ax.bar(x + width/2, test_vals, width, label="Test (hold-out)", color="#e74c3c", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
    fig.suptitle("CV Mean vs Hold-out Test Evaluation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "03_cv_vs_test_comparison")

    # ── 3. Boxplot distribusi R² per fold ──
    fig, ax = plt.subplots(figsize=(10, 6))
    r2_data = [results[n]["r2_scores"] for n in names]
    bp = ax.boxplot(r2_data, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2), notch=True)
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("R² Score")
    ax.set_title(
        f"Distribusi R² per Fold\n"
        f"(Repeated K-Fold: {N_SPLITS}×{N_REPEATS} = {N_SPLITS*N_REPEATS} fold)",
        fontweight="bold"
    )
    ax.axhline(0.9, color="green", linestyle="--", alpha=0.5, label="R²=0.90")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "04_r2_distribution_boxplot")

    # ── 4. Violin plot R² per fold ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.violinplot(r2_data, positions=range(len(names)), showmedians=True)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("R² Score")
    ax.set_title(
        f"Distribusi R² per Fold — Violin Plot\n"
        f"(Repeated K-Fold: {N_SPLITS}×{N_REPEATS} = {N_SPLITS*N_REPEATS} fold)",
        fontweight="bold"
    )
    fig.tight_layout()
    save_fig(fig, "05_r2_distribution_violin")

    # ── 5. R² per fold — line plot semua iterasi ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    repeat_colors = sns.color_palette("tab10", N_REPEATS)

    for ax, (name, res) in zip(axes, results.items()):
        scores   = res["r2_scores"]
        fold_ids = np.arange(1, len(scores) + 1)

        for i, (fid, sc) in enumerate(zip(fold_ids, scores)):
            repeat_idx = i // N_SPLITS
            label = f"Repeat {repeat_idx + 1}" if i % N_SPLITS == 0 else None
            ax.scatter(fid, sc, color=repeat_colors[repeat_idx], s=40, zorder=3, label=label)

        ax.plot(fold_ids, scores, color="gray", alpha=0.4, linewidth=1)
        ax.axhline(scores.mean(), color="red", linestyle="--",
                   linewidth=1.5, label=f"CV Mean={scores.mean():.4f}")
        ax.fill_between(
            fold_ids,
            scores.mean() - scores.std(),
            scores.mean() + scores.std(),
            alpha=0.1, color="red", label=f"±Std={scores.std():.4f}"
        )

        # Separator antar repeat
        for r in range(1, N_REPEATS):
            ax.axvline(r * N_SPLITS + 0.5, color="black",
                       linestyle=":", alpha=0.3, linewidth=1)

        ax.set_xlabel("Fold ke-")
        ax.set_ylabel("R² Score")
        ax.set_title(f"{name}", fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        # Anotasi repeat
        for r in range(N_REPEATS):
            mid = r * N_SPLITS + N_SPLITS / 2 + 0.5
            ax.text(mid, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else scores.min() - 0.001,
                    f"R{r+1}", ha="center", va="bottom", fontsize=8, color="gray")

    fig.suptitle(
        f"R² Score per Fold — Repeated K-Fold\n"
        f"({N_SPLITS} splits × {N_REPEATS} repeats = {N_SPLITS*N_REPEATS} fold)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    save_fig(fig, "06_r2_per_fold_all_repeats")

    # ── 6. Train vs Val R² (overfit check) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35
    train_r2 = [results[n]["r2_train"].mean() for n in names]
    val_r2   = [results[n]["cv_r2"]            for n in names]
    ax.bar(x - width/2, train_r2, width, label="Train R²",      color="#2ecc71", alpha=0.85)
    ax.bar(x + width/2, val_r2,   width, label="Val R² (CV)",   color="#3498db", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("R² Score")
    ax.set_title("Train R² vs Validation R² (Overfit Check)", fontweight="bold")
    ax.legend()
    for i, (tr, vl) in enumerate(zip(train_r2, val_r2)):
        gap = tr - vl
        ax.text(i, max(tr, vl) + 0.005, f"gap={gap:.4f}", ha="center", fontsize=9,
                color="red" if gap > 0.05 else "green")
    fig.tight_layout()
    save_fig(fig, "07_train_vs_val_r2")

    # ── 7. Heatmap stabilitas R² per fold × model ──
    fig, ax = plt.subplots(figsize=(14, 5))
    r2_matrix = np.array([results[n]["r2_scores"] for n in names])   # (n_models, n_folds)
    fold_labels = []
    for rep in range(1, N_REPEATS + 1):
        for fold in range(1, N_SPLITS + 1):
            fold_labels.append(f"R{rep}F{fold}")
    sns.heatmap(r2_matrix, ax=ax, xticklabels=fold_labels,
                yticklabels=names, annot=False, cmap="YlGn",
                cbar_kws={"label": "R² Score"}, linewidths=0.3)
    ax.set_xlabel("Fold (R=Repeat, F=Fold)")
    ax.set_title(
        f"Heatmap Stabilitas R² — Semua {N_SPLITS*N_REPEATS} Fold\n"
        f"(Repeated K-Fold: {N_SPLITS}×{N_REPEATS})",
        fontweight="bold"
    )
    plt.xticks(rotation=90, fontsize=7)
    fig.tight_layout()
    save_fig(fig, "08_r2_stability_heatmap")


# ── 9. Feature Importance (Random Forest) ────────────────────────────────────

def plot_feature_importance(results: dict, feature_names: list,
                            mi_series: pd.Series) -> None:
    section("9. Feature Importance vs MI Score")

    rf_model    = results["Random Forest"]["model"]
    importances = rf_model.feature_importances_
    imp_series  = pd.Series(importances, index=feature_names).sort_values(ascending=False)

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

    # MI vs RF grouped bar
    x     = np.arange(len(compare_df))
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
    save_fig(fig, "09_feature_importance_vs_mi")

    print("\n  Perbandingan ranking MI vs RF Importance:")
    mi_rank  = mi_series[feature_names].rank(ascending=False).astype(int)
    imp_rank = imp_series.rank(ascending=False).astype(int)
    rank_df  = pd.DataFrame({
        "MI Rank":   mi_rank,
        "RF Rank":   imp_rank,
        "Rank Diff": (mi_rank - imp_rank).abs()
    }).sort_values("MI Rank")
    print(rank_df.to_string())


# ── 10. Ringkasan ─────────────────────────────────────────────────────────────

def print_summary(results: dict, mi_series: pd.Series,
                  selected_features: list) -> None:
    section("Ringkasan Akhir")

    # Best by CV R²
    best_name = max(results, key=lambda n: results[n]["cv_r2"])
    best      = results[best_name]

    print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │           TRACK 1 – FLOOD PREDICTION SUMMARY            │
  ├──────────────────────────────────────────────────────────┤
  │  Metode Feature Selection : Mutual Information           │
  │  Threshold MI             : {str(MI_THRESHOLD):<28}│
  │  Fitur terpilih           : {len(selected_features):<28}│
  ├──────────────────────────────────────────────────────────┤
  │  CV Strategy  : RepeatedKFold                            │
  │  n_splits     : {N_SPLITS:<44}│
  │  n_repeats    : {N_REPEATS:<44}│
  │  Total fold   : {N_SPLITS*N_REPEATS:<44}│
  ├──────────────────────────────────────────────────────────┤
  │  Model Terbaik (CV R²)    : {best_name:<28}│
  │  CV R² (mean ± std)       : {best['cv_r2']:.4f} ± {best['cv_r2_std']:.4f}               │
  │  CV RMSE (mean)           : {best['cv_rmse']:<28.6f}│
  │  CV MAE  (mean)           : {best['cv_mae']:<28.6f}│
  │  Test R² (hold-out)       : {best['test_r2']:<28.4f}│
  │  Test RMSE                : {best['test_rmse']:<28.6f}│
  │  Test MAE                 : {best['test_mae']:<28.6f}│
  ├──────────────────────────────────────────────────────────┤
  │  Semua Model – Ringkasan CV R²                           │""")

    for name, res in sorted(results.items(), key=lambda x: -x[1]["cv_r2"]):
        tag = " ← terbaik" if name == best_name else ""
        print(f"  │    • {name:<30} {res['cv_r2']:.4f} ± {res['cv_r2_std']:.4f}{tag:<12}│")

    print(f"  ├──────────────────────────────────────────────────────────┤")
    print(f"  │  Top-3 Fitur (MI Score)                                  │")
    for feat, score in mi_series.head(3).items():
        short = feat[:38] + "…" if len(feat) > 38 else feat
        print(f"  │    • {short:<40} {score:.4f}  │")

    print(f"  └──────────────────────────────────────────────────────────┘")
    print(f"\n  Semua plot disimpan ke: {OUTPUT_DIR.resolve()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Split fitur & target
    X, y = split_features_target(df)

    # 3. Train-test split (hold-out)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # 4. Hitung Mutual Information (dari train set agar tidak bocor)
    mi_series = compute_mutual_information(X_train, y_train)

    # 5. Feature selection berdasarkan MI
    X_train_sel = select_features(X_train, mi_series, threshold=MI_THRESHOLD)
    X_test_sel  = X_test[X_train_sel.columns]  # Terapkan kolom yang sama ke test set

    # 6. Setup Repeated K-Fold
    rkf = setup_kfold()

    # 7. Training & evaluasi semua model
    results = train_models(X_train_sel, X_test_sel, y_train, y_test, rkf)

    # 8. Visualisasi evaluasi
    plot_evaluation(results)

    # 9. Feature importance vs MI
    plot_feature_importance(results, X_train_sel.columns.tolist(), mi_series)

    # 10. Ringkasan
    print_summary(results, mi_series, X_train_sel.columns.tolist())


if __name__ == "__main__":
    main()