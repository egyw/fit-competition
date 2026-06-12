"""
Track 1 – Flood Prediction
LightGBM + 3 Skenario Fitur + Repeated K-Fold Cross Validation
================================================================
Dataset      : flood.csv (50.000 baris, 20 fitur, 1 target)
Task         : Regression → prediksi FloodProbability
CV Strategy  : RepeatedKFold(n_splits=10, n_repeats=3) = 30 total fold
Skenario     :
  1. Minimal          →  8 fitur (MI top features)
  2. Balanced         → 11 fitur (top-8 + 3 tambahan)
  3. Maksimum Akurasi → 20 fitur (semua fitur)

Run:
    pip install lightgbm
    python lightgbm_flood.py

Output disimpan ke ./flood_lgbm_outputs/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import lightgbm as lgb

from pathlib import Path
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ── Konfigurasi ───────────────────────────────────────────────────────────────

DATA_PATH  = "data/raw/flood.csv"
OUTPUT_DIR = Path("flood_lgbm_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL   = "FloodProbability"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Repeated K-Fold
N_SPLITS  = 10
N_REPEATS = 3
# Total = 30 fold

# ── Definisi 3 Skenario Fitur ─────────────────────────────────────────────────

SCENARIOS = {
    "Minimal (8 fitur)": [
        "MonsoonIntensity",
        "Encroachments",
        "TopographyDrainage",
        "DamsQuality",
        "IneffectiveDisasterPreparedness",
        "ClimateChange",
        "InadequatePlanning",
        "Watersheds",
    ],
    "Balanced (11 fitur)": [
        # Top-8 Minimal
        "MonsoonIntensity",
        "Encroachments",
        "TopographyDrainage",
        "DamsQuality",
        "IneffectiveDisasterPreparedness",
        "ClimateChange",
        "InadequatePlanning",
        "Watersheds",
        # +3 tambahan
        "CoastalVulnerability",
        "Deforestation",
        "Siltation",
    ],
    "Maksimum Akurasi (20 fitur)": None,  # None = semua fitur
}

# ── LightGBM Hyperparameter ───────────────────────────────────────────────────
# Satu set parameter yang cukup kuat untuk ketiga skenario.
# Anda bisa tuning per skenario jika perlu.

LGBM_PARAMS = {
    "objective":       "regression",
    "metric":          "rmse",
    "boosting_type":   "gbdt",
    "n_estimators":    500,
    "learning_rate":   0.05,
    "num_leaves":      63,
    "max_depth":       -1,          # -1 = tidak dibatasi
    "min_child_samples": 20,
    "subsample":       0.8,
    "subsample_freq":  1,
    "colsample_bytree": 0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      0.1,
    "n_jobs":          -1,
    "random_state":    RANDOM_STATE,
    "verbose":         -1,
}

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
SCENARIO_COLORS = {
    "Minimal (8 fitur)":            "#e74c3c",
    "Balanced (11 fitur)":          "#f39c12",
    "Maksimum Akurasi (20 fitur)":  "#2ecc71",
}


# ── Helper ────────────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot disimpan → {path}")


def section(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ── 1. Load Data ──────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    section("1. Load Data")
    df = pd.read_csv(path)
    print(f"  Shape          : {df.shape}")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    print(f"  Target range   : {df[TARGET_COL].min():.3f} – {df[TARGET_COL].max():.3f}")
    print(f"  Target mean    : {df[TARGET_COL].mean():.3f}")
    print(f"  Target std     : {df[TARGET_COL].std():.4f}")
    return df


# ── 2. Train-Test Split ───────────────────────────────────────────────────────

def split_train_test(df: pd.DataFrame):
    section("2. Train-Test Split (Hold-out)")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train set : {X_train.shape[0]:,} baris ({int((1-TEST_SIZE)*100)}%)")
    print(f"  Test set  : {X_test.shape[0]:,} baris  ({int(TEST_SIZE*100)}%)")
    print(f"  ⚠  Test set HANYA digunakan untuk evaluasi final (hold-out).")
    print(f"     Repeated K-Fold dijalankan di atas train set saja.")
    return X_train, X_test, y_train, y_test


# ── 3. Skenario Feature Sets ──────────────────────────────────────────────────

def resolve_scenarios(X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    section("3. Skenario Fitur")
    all_features = X_train.columns.tolist()

    resolved = {}
    for name, feats in SCENARIOS.items():
        if feats is None:
            feats = all_features
        resolved[name] = {
            "X_train": X_train[feats],
            "X_test":  X_test[feats],
            "features": feats,
        }
        print(f"\n  [{name}]")
        print(f"    Jumlah fitur : {len(feats)}")
        print(f"    Fitur        : {feats}")

    return resolved


# ── 4. Setup Repeated K-Fold ──────────────────────────────────────────────────

def setup_kfold() -> RepeatedKFold:
    section("4. Repeated K-Fold Setup")
    rkf = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )
    train_n = int(50000 * (1 - TEST_SIZE))
    print(f"  Strategi       : RepeatedKFold")
    print(f"  n_splits       : {N_SPLITS}  (fold per repeat)")
    print(f"  n_repeats      : {N_REPEATS}  (jumlah pengulangan)")
    print(f"  Total iterasi  : {N_SPLITS * N_REPEATS} fold")
    print(f"  Random state   : {RANDOM_STATE}")
    print(f"  Train per fold : ~{int(train_n * (N_SPLITS-1)/N_SPLITS):,} baris")
    print(f"  Val per fold   : ~{int(train_n / N_SPLITS):,} baris")
    return rkf


# ── 5. Training & Evaluasi per Skenario ──────────────────────────────────────

def run_scenario(scenario_name: str, scenario_data: dict,
                 y_train: pd.Series, y_test: pd.Series,
                 rkf: RepeatedKFold) -> dict:
    """
    Menjalankan Repeated K-Fold CV + hold-out evaluation
    untuk satu skenario fitur.
    """
    X_tr = scenario_data["X_train"].values
    X_te = scenario_data["X_test"].values
    y_tr = y_train.values
    y_te = y_test.values

    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    # ── Repeated K-Fold CV ──
    scoring = {
        "r2":   "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae":  "neg_mean_absolute_error",
    }
    cv_result = cross_validate(
        model, X_tr, y_tr,
        cv=rkf,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True
    )

    r2_cv    =  cv_result["test_r2"]
    rmse_cv  = -cv_result["test_rmse"]
    mae_cv   = -cv_result["test_mae"]
    r2_train =  cv_result["train_r2"]

    # ── Hold-out Test Evaluation ──
    model.fit(X_tr, y_tr)
    y_pred    = model.predict(X_te)
    test_r2   = r2_score(y_te, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    test_mae  = mean_absolute_error(y_te, y_pred)

    print(f"\n  ── {scenario_name} ──")
    print(f"  Fitur: {len(scenario_data['features'])}")
    print(f"  [Repeated K-Fold CV — {N_SPLITS*N_REPEATS} fold]")
    print(f"  {'Metrik':<8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*52}")
    print(f"  {'R²':<8} {r2_cv.mean():>10.6f} {r2_cv.std():>10.6f} {r2_cv.min():>10.6f} {r2_cv.max():>10.6f}")
    print(f"  {'RMSE':<8} {rmse_cv.mean():>10.6f} {rmse_cv.std():>10.6f} {rmse_cv.min():>10.6f} {rmse_cv.max():>10.6f}")
    print(f"  {'MAE':<8} {mae_cv.mean():>10.6f} {mae_cv.std():>10.6f} {mae_cv.min():>10.6f} {mae_cv.max():>10.6f}")
    print(f"  Train R² mean : {r2_train.mean():.6f}  |  overfit gap : {r2_train.mean()-r2_cv.mean():.6f}")
    print(f"  [Hold-out Test]  R²={test_r2:.6f}  RMSE={test_rmse:.6f}  MAE={test_mae:.6f}")

    return {
        "model":        model,
        "features":     scenario_data["features"],
        "n_features":   len(scenario_data["features"]),
        # CV
        "r2_scores":    r2_cv,
        "rmse_scores":  rmse_cv,
        "mae_scores":   mae_cv,
        "r2_train":     r2_train,
        "cv_r2":        r2_cv.mean(),
        "cv_r2_std":    r2_cv.std(),
        "cv_rmse":      rmse_cv.mean(),
        "cv_mae":       mae_cv.mean(),
        # Hold-out
        "test_r2":      test_r2,
        "test_rmse":    test_rmse,
        "test_mae":     test_mae,
        # For hold-out residuals / prediction
        "y_pred":       y_pred,
    }


def train_all_scenarios(scenarios: dict, y_train: pd.Series,
                        y_test: pd.Series, rkf: RepeatedKFold) -> dict:
    section("5. Training & Evaluasi — Semua Skenario LightGBM")
    print(f"  Model          : LightGBM (LGBMRegressor)")
    print(f"  n_estimators   : {LGBM_PARAMS['n_estimators']}")
    print(f"  learning_rate  : {LGBM_PARAMS['learning_rate']}")
    print(f"  num_leaves     : {LGBM_PARAMS['num_leaves']}")
    print(f"  CV             : RepeatedKFold({N_SPLITS}×{N_REPEATS} = {N_SPLITS*N_REPEATS} fold)\n")

    results = {}
    for name, data in scenarios.items():
        results[name] = run_scenario(name, data, y_train, y_test, rkf)

    return results


# ── 6. Visualisasi ────────────────────────────────────────────────────────────

def plot_scenario_comparison(results: dict, y_test: pd.Series) -> None:
    section("6. Visualisasi Perbandingan Skenario")

    names   = list(results.keys())
    colors  = [SCENARIO_COLORS[n] for n in names]
    palette = colors

    # ── Plot 1: Bar chart metrik CV ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("cv_r2",   "cv_r2_std", "Mean CV R² ± Std",  "R²"),
        ("cv_rmse", None,        "Mean CV RMSE",       "RMSE"),
        ("cv_mae",  None,        "Mean CV MAE",        "MAE"),
    ]
    for ax, (key, std_key, title, ylabel) in zip(axes, metrics):
        vals  = [results[n][key] for n in names]
        stds  = [results[n][std_key] for n in names] if std_key else None
        bars  = ax.bar(names, vals, color=palette, edgecolor="white",
                       yerr=stds, capsize=6 if stds else 0)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(names, rotation=18, ha="right")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    fig.suptitle(
        f"Perbandingan Skenario Fitur — LightGBM\n"
        f"Repeated K-Fold: {N_SPLITS} splits × {N_REPEATS} repeats = {N_SPLITS*N_REPEATS} fold",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    save_fig(fig, "01_scenario_comparison_cv_metrics")

    # ── Plot 2: CV vs Hold-out Test ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x     = np.arange(len(names))
    width = 0.35
    for ax, (cv_key, test_key, ylabel, title) in zip(axes, [
        ("cv_r2",   "test_r2",   "R²",   "R² : CV vs Test"),
        ("cv_rmse", "test_rmse", "RMSE", "RMSE : CV vs Test"),
        ("cv_mae",  "test_mae",  "MAE",  "MAE : CV vs Test"),
    ]):
        cv_v   = [results[n][cv_key]   for n in names]
        test_v = [results[n][test_key] for n in names]
        ax.bar(x - width/2, cv_v,   width, label="CV Mean",        color="#3498db", alpha=0.85)
        ax.bar(x + width/2, test_v, width, label="Hold-out Test",  color="#e74c3c", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        for i, (cv, te) in enumerate(zip(cv_v, test_v)):
            diff = abs(cv - te)
            ax.text(i, max(cv, te) + max(cv_v + test_v) * 0.01,
                    f"Δ{diff:.4f}", ha="center", fontsize=8, color="gray")
    fig.suptitle("CV Mean vs Hold-out Test Evaluation per Skenario", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "02_cv_vs_test_per_scenario")

    # ── Plot 3: Boxplot distribusi R² per fold × skenario ──
    fig, ax = plt.subplots(figsize=(12, 6))
    data_box = [results[n]["r2_scores"] for n in names]
    bp = ax.boxplot(data_box, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2), notch=True)
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylabel("R² Score")
    ax.set_title(
        f"Distribusi R² per Fold — LightGBM\n"
        f"(Repeated K-Fold: {N_SPLITS}×{N_REPEATS} = {N_SPLITS*N_REPEATS} fold)",
        fontweight="bold"
    )
    ax.axhline(0.9, color="navy", linestyle="--", alpha=0.5, label="R²=0.90")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "03_r2_boxplot_per_scenario")

    # ── Plot 4: R² per fold semua iterasi (3 panel) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    repeat_colors = sns.color_palette("tab10", N_REPEATS)
    for ax, (name, res) in zip(axes, results.items()):
        scores   = res["r2_scores"]
        fold_ids = np.arange(1, len(scores) + 1)
        for i, (fid, sc) in enumerate(zip(fold_ids, scores)):
            rep_idx = i // N_SPLITS
            label   = f"Repeat {rep_idx+1}" if i % N_SPLITS == 0 else None
            ax.scatter(fid, sc, color=repeat_colors[rep_idx], s=40, zorder=3, label=label)
        ax.plot(fold_ids, scores, color="gray", alpha=0.3, linewidth=1)
        ax.axhline(scores.mean(), color="red", linestyle="--",
                   linewidth=1.5, label=f"Mean={scores.mean():.4f}")
        ax.fill_between(fold_ids,
                        scores.mean() - scores.std(),
                        scores.mean() + scores.std(),
                        alpha=0.1, color="red", label=f"±Std={scores.std():.4f}")
        for r in range(1, N_REPEATS):
            ax.axvline(r * N_SPLITS + 0.5, color="black", linestyle=":", alpha=0.3)
        ax.set_xlabel("Fold ke-")
        ax.set_ylabel("R² Score")
        ax.set_title(name, fontweight="bold", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(
        f"R² Score per Fold — Repeated K-Fold ({N_SPLITS}×{N_REPEATS})",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    save_fig(fig, "04_r2_per_fold_all_repeats")

    # ── Plot 5: Heatmap stabilitas semua fold ──
    fig, ax = plt.subplots(figsize=(18, 4))
    r2_matrix = np.array([results[n]["r2_scores"] for n in names])
    fold_labels = [f"R{r+1}F{f+1}" for r in range(N_REPEATS) for f in range(N_SPLITS)]
    sns.heatmap(r2_matrix, ax=ax, xticklabels=fold_labels,
                yticklabels=names, annot=True, fmt=".3f",
                cmap="YlGn", cbar_kws={"label": "R² Score"},
                linewidths=0.3, annot_kws={"size": 7})
    ax.set_xlabel(f"Fold (R=Repeat, F=Fold) — total {N_SPLITS*N_REPEATS} fold")
    ax.set_title(
        f"Heatmap Stabilitas R² — LightGBM\n"
        f"Repeated K-Fold {N_SPLITS}×{N_REPEATS} = {N_SPLITS*N_REPEATS} fold",
        fontweight="bold"
    )
    plt.xticks(rotation=90, fontsize=7)
    fig.tight_layout()
    save_fig(fig, "05_r2_stability_heatmap")

    # ── Plot 6: Train vs Val R² (overfit check) ──
    fig, ax = plt.subplots(figsize=(11, 5))
    x     = np.arange(len(names))
    width = 0.35
    tr_r2 = [results[n]["r2_train"].mean() for n in names]
    vl_r2 = [results[n]["cv_r2"]           for n in names]
    ax.bar(x - width/2, tr_r2, width, label="Train R²",    color="#2ecc71", alpha=0.85)
    ax.bar(x + width/2, vl_r2, width, label="Val R² (CV)", color="#3498db", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylabel("R² Score")
    ax.set_title("Train R² vs Validation R² — Overfit Check", fontweight="bold")
    ax.legend()
    for i, (tr, vl) in enumerate(zip(tr_r2, vl_r2)):
        gap = tr - vl
        ax.text(i, max(tr, vl) + 0.003, f"gap={gap:.4f}", ha="center", fontsize=9,
                color="red" if gap > 0.05 else "green")
    fig.tight_layout()
    save_fig(fig, "06_overfit_check")

    # ── Plot 7: Residual plot hold-out per skenario ──
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    y_te = y_test.values
    for ax, (name, res) in zip(axes, results.items()):
        y_pred = res["y_pred"]
        resid  = y_te - y_pred
        ax.scatter(y_pred, resid, alpha=0.3, s=8, color=SCENARIO_COLORS[name])
        ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Predicted FloodProbability")
        ax.set_ylabel("Residual (actual − pred)")
        ax.set_title(f"{name}\nTest R²={res['test_r2']:.4f}", fontweight="bold", fontsize=9)
    fig.suptitle("Residual Plot — Hold-out Test Set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "07_residual_plots")

    # ── Plot 8: Actual vs Predicted per skenario ──
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (name, res) in zip(axes, results.items()):
        y_pred = res["y_pred"]
        ax.scatter(y_te, y_pred, alpha=0.3, s=8, color=SCENARIO_COLORS[name])
        lo = min(y_te.min(), y_pred.min())
        hi = max(y_te.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual FloodProbability")
        ax.set_ylabel("Predicted FloodProbability")
        ax.set_title(f"{name}\nRMSE={res['test_rmse']:.4f}  MAE={res['test_mae']:.4f}",
                     fontweight="bold", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Actual vs Predicted — Hold-out Test Set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "08_actual_vs_predicted")


# ── 7. Feature Importance per Skenario ───────────────────────────────────────

def plot_feature_importance(results: dict) -> None:
    section("7. Feature Importance per Skenario (LightGBM)")

    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    for ax, (name, res) in zip(axes, results.items()):
        model     = res["model"]
        features  = res["features"]
        imp       = model.feature_importances_
        imp_s     = pd.Series(imp, index=features).sort_values(ascending=True)
        colors    = [SCENARIO_COLORS[name]] * len(imp_s)
        ax.barh(imp_s.index, imp_s.values, color=colors, edgecolor="white")
        ax.set_xlabel("Importance (gain)")
        ax.set_title(name, fontweight="bold", fontsize=9)
        for i, (feat, val) in enumerate(zip(imp_s.index, imp_s.values)):
            ax.text(val + imp_s.max() * 0.01, i, f"{val:.0f}", va="center", fontsize=7)

    fig.suptitle("LightGBM Feature Importance (Gain) per Skenario",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "09_feature_importance_per_scenario")

    # Tabel ranking fitur yang muncul di beberapa skenario
    print("\n  Ranking Feature Importance per Skenario:")
    for name, res in results.items():
        model    = res["model"]
        features = res["features"]
        imp      = pd.Series(model.feature_importances_, index=features) \
                     .sort_values(ascending=False)
        print(f"\n  [{name}]")
        print(f"  {'Rank':<5} {'Fitur':<40} {'Importance':>12}")
        print(f"  {'-'*60}")
        for rank, (feat, val) in enumerate(imp.items(), 1):
            print(f"  {rank:<5} {feat:<40} {val:>12.1f}")


# ── 8. Ringkasan Akhir ────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    section("Ringkasan Akhir")

    best_name = max(results, key=lambda n: results[n]["cv_r2"])

    print(f"""
  ┌────────────────────────────────────────────────────────────┐
  │         TRACK 1 – FLOOD PREDICTION  (LightGBM)            │
  ├────────────────────────────────────────────────────────────┤
  │  Model          : LightGBM (LGBMRegressor)                 │
  │  n_estimators   : {LGBM_PARAMS['n_estimators']:<42}│
  │  learning_rate  : {LGBM_PARAMS['learning_rate']:<42}│
  │  num_leaves     : {LGBM_PARAMS['num_leaves']:<42}│
  ├────────────────────────────────────────────────────────────┤
  │  CV Strategy    : RepeatedKFold                            │
  │  n_splits       : {N_SPLITS:<42}│
  │  n_repeats      : {N_REPEATS:<42}│
  │  Total fold     : {N_SPLITS*N_REPEATS:<42}│
  ├────────────────────────────────────────────────────────────┤
  │  Perbandingan Skenario                                     │
  │  {'Skenario':<30} {'Fitur':>5} {'CV R²':>8} {'±Std':>7} {'Test R²':>9}│
  │  {'-'*60}│""")

    for name, res in results.items():
        marker = " ✓" if name == best_name else "  "
        print(f"  │  {name:<30} {res['n_features']:>5} {res['cv_r2']:>8.4f} {res['cv_r2_std']:>7.4f} {res['test_r2']:>9.4f}{marker}│")

    print(f"  ├────────────────────────────────────────────────────────────┤")
    best = results[best_name]
    print(f"  │  Skenario Terbaik (CV R²): {best_name:<34}│")
    print(f"  │  CV R²  (mean ± std)     : {best['cv_r2']:.4f} ± {best['cv_r2_std']:.4f}                   │")
    print(f"  │  CV RMSE (mean)          : {best['cv_rmse']:<34.6f}│")
    print(f"  │  Test R² (hold-out)      : {best['test_r2']:<34.4f}│")
    print(f"  │  Test RMSE               : {best['test_rmse']:<34.6f}│")
    print(f"  └────────────────────────────────────────────────────────────┘")
    print(f"\n  Semua plot disimpan ke: {OUTPUT_DIR.resolve()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Hold-out split
    X_train, X_test, y_train, y_test = split_train_test(df)

    # 3. Resolusi skenario fitur
    scenarios = resolve_scenarios(X_train, X_test)

    # 4. Setup Repeated K-Fold
    rkf = setup_kfold()

    # 5. Training & evaluasi semua skenario
    results = train_all_scenarios(scenarios, y_train, y_test, rkf)

    # 6. Visualisasi perbandingan skenario
    plot_scenario_comparison(results, y_test)

    # 7. Feature importance per skenario
    plot_feature_importance(results)

    # 8. Ringkasan
    print_summary(results)


if __name__ == "__main__":
    main()