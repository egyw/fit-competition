# ==========================================================
# RIDGE REGRESSION WITH 3 FEATURE SCENARIOS
# Repeated K-Fold CV (10 folds × 3 repeats)
# Includes Prediction vs Actual Plot
# ==========================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    RepeatedKFold,
    cross_validate,
    cross_val_predict
)

# ==========================================================
# PATHS
# ==========================================================

DATA_PATH = r"C:\CHRISTOPHER\LOMBA\FIT\CODE\data\raw\flood.csv"

OUTPUT_DIR = r"C:\CHRISTOPHER\LOMBA\FIT\CODE\results\ridge"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

TARGET = "FloodProbability"

# ==========================================================
# FEATURE SCENARIOS
# ==========================================================

features_8 = [
    "MonsoonIntensity",
    "Encroachments",
    "TopographyDrainage",
    "DamsQuality",
    "IneffectiveDisasterPreparedness",
    "ClimateChange",
    "InadequatePlanning",
    "Watersheds"
]

features_11 = [
    "MonsoonIntensity",
    "Encroachments",
    "TopographyDrainage",
    "DamsQuality",
    "IneffectiveDisasterPreparedness",
    "ClimateChange",
    "InadequatePlanning",
    "Watersheds",
    "CoastalVulnerability",
    "Deforestation",
    "Siltation"
]

features_20 = [col for col in df.columns if col != TARGET]

scenarios = {
    "8 Features": features_8,
    "11 Features": features_11,
    "20 Features": features_20
}

# ==========================================================
# REPEATED K-FOLD
# ==========================================================

rkf = RepeatedKFold(
    n_splits=10,
    n_repeats=3,
    random_state=42
)

# ==========================================================
# STORAGE
# ==========================================================

summary_results = []

r2_distributions = {}
mae_distributions = {}
rmse_distributions = {}

# ==========================================================
# EVALUATION LOOP
# ==========================================================

for scenario_name, feature_list in scenarios.items():

    print(f"\nRunning: {scenario_name}")

    X = df[feature_list]
    y = df[TARGET]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    # ------------------------------------------------------
    # Cross Validation Metrics
    # ------------------------------------------------------

    scores = cross_validate(
        model,
        X,
        y,
        cv=rkf,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error"
        },
        n_jobs=-1
    )

    r2_scores = scores["test_r2"]
    mae_scores = -scores["test_mae"]
    rmse_scores = -scores["test_rmse"]

    r2_distributions[scenario_name] = r2_scores
    mae_distributions[scenario_name] = mae_scores
    rmse_distributions[scenario_name] = rmse_scores

    summary_results.append({
        "Scenario": scenario_name,
        "Num Features": len(feature_list),

        "Mean R2": np.mean(r2_scores),
        "Std R2": np.std(r2_scores),

        "Mean MAE": np.mean(mae_scores),
        "Std MAE": np.std(mae_scores),

        "Mean RMSE": np.mean(rmse_scores),
        "Std RMSE": np.std(rmse_scores)
    })

    # ------------------------------------------------------
    # Prediction vs Actual
    # ------------------------------------------------------

    print("Generating CV predictions...")

    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=10,
        n_jobs=-1
    )

    pred_df = pd.DataFrame({
        "Actual": y,
        "Predicted": y_pred
    })

    scenario_file = scenario_name.lower().replace(" ", "_")

    pred_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{scenario_file}_predictions.csv"
        ),
        index=False
    )

    # ------------------------------------------------------
    # Prediction vs Actual Plot
    # ------------------------------------------------------

    plt.figure(figsize=(8, 8))

    plt.scatter(
        y,
        y_pred,
        alpha=0.4
    )

    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linewidth=2
    )

    plt.xlabel("Actual Flood Probability")
    plt.ylabel("Predicted Flood Probability")

    plt.title(
        f"Prediction vs Actual\n{scenario_name}"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{scenario_file}_prediction_vs_actual.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ==========================================================
# SUMMARY RESULTS
# ==========================================================

results_df = pd.DataFrame(summary_results)

print("\n")
print("=" * 80)
print(results_df)
print("=" * 80)

# ==========================================================
# SAVE RESULTS TABLE
# ==========================================================

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ridge_metrics.csv"
    ),
    index=False
)

results_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "ridge_metrics.xlsx"
    ),
    index=False
)

# ==========================================================
# R² COMPARISON
# ==========================================================

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Scenario"],
    results_df["Mean R2"]
)

plt.title("Mean R² Comparison")
plt.ylabel("R²")
plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ridge_r2_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# MAE COMPARISON
# ==========================================================

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Scenario"],
    results_df["Mean MAE"]
)

plt.title("Mean MAE Comparison")
plt.ylabel("MAE")
plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ridge_mae_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# RMSE COMPARISON
# ==========================================================

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Scenario"],
    results_df["Mean RMSE"]
)

plt.title("Mean RMSE Comparison")
plt.ylabel("RMSE")
plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ridge_rmse_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# R² BOXPLOT
# ==========================================================

plt.figure(figsize=(10, 6))

plt.boxplot([
    r2_distributions["8 Features"],
    r2_distributions["11 Features"],
    r2_distributions["20 Features"]
])

plt.xticks(
    [1, 2, 3],
    ["8 Features", "11 Features", "20 Features"]
)

plt.ylabel("R²")
plt.title("R² Distribution Across Repeated K-Folds")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ridge_r2_boxplot.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# MAE BOXPLOT
# ==========================================================

plt.figure(figsize=(10, 6))

plt.boxplot([
    mae_distributions["8 Features"],
    mae_distributions["11 Features"],
    mae_distributions["20 Features"]
])

plt.xticks(
    [1, 2, 3],
    ["8 Features", "11 Features", "20 Features"]
)

plt.ylabel("MAE")
plt.title("MAE Distribution Across Repeated K-Folds")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ridge_mae_boxplot.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# RMSE BOXPLOT
# ==========================================================

plt.figure(figsize=(10, 6))

plt.boxplot([
    rmse_distributions["8 Features"],
    rmse_distributions["11 Features"],
    rmse_distributions["20 Features"]
])

plt.xticks(
    [1, 2, 3],
    ["8 Features", "11 Features", "20 Features"]
)

plt.ylabel("RMSE")
plt.title("RMSE Distribution Across Repeated K-Folds")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ridge_rmse_boxplot.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# SAVE FOLD RESULTS
# ==========================================================

fold_results = []

for scenario in r2_distributions.keys():

    for i in range(len(r2_distributions[scenario])):

        fold_results.append({
            "Scenario": scenario,
            "Fold": i + 1,
            "R2": r2_distributions[scenario][i],
            "MAE": mae_distributions[scenario][i],
            "RMSE": rmse_distributions[scenario][i]
        })

fold_df = pd.DataFrame(fold_results)

fold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ridge_fold_results.csv"
    ),
    index=False
)

# ==========================================================
# COMPLETE
# ==========================================================

print("\nAll files saved successfully.")

print(f"\nOutput folder:\n{OUTPUT_DIR}")