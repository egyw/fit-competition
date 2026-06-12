from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "FloodProbability"
FIGURE_FILENAMES = {
    "histograms": "1_histograms.png",
    "boxplots": "2_boxplots.png",
    "correlation_heatmap": "3_correlation_heatmap.png",
    "target_distribution": "4_target_distribution.png",
    "target_correlations": "5_target_correlations.png",
    "scatter_top_features": "6_scatter_top_features.png",
    "pairplot_top_features": "7_pairplot_top_features.png",
    "rf_feature_importance": "8_rf_feature_importance.png",
    "actual_vs_predicted": "9_actual_vs_predicted.png",
    "residual_distribution": "10_residual_distribution.png",
}


def configure_logging() -> logging.Logger:
    """Configure console logging for the EDA pipeline."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("track_1_eda")


def get_project_root() -> Path:
    """Resolve the project root relative to this file."""
    return Path(__file__).resolve().parents[2]


def get_dataset_path(project_root: Path) -> Path:
    """Return the raw flood dataset path."""
    return project_root / "data" / "raw" / "flood.csv"


def get_output_dir(project_root: Path) -> Path:
    """Return the figure output directory and create it if needed."""
    output_dir = project_root / "reports" / "figures" / "track_1"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load the flood dataset from disk."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")
    return df


def setup_plot_style() -> None:
    """Set a consistent seaborn/matplotlib style for portfolio-quality figures."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify columns into numeric, categorical, datetime, boolean, and text."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    boolean = df.select_dtypes(include=["bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    categorical: list[str] = []
    text: list[str] = []
    for column in object_cols:
        unique_ratio = df[column].nunique(dropna=True) / max(df[column].dropna().shape[0], 1)
        if unique_ratio <= 0.2 or df[column].nunique(dropna=True) <= 20:
            categorical.append(column)
        else:
            text.append(column)

    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_cols,
        "boolean": boolean,
        "text": text,
    }


def print_dataset_summary(df: pd.DataFrame, logger: logging.Logger) -> dict[str, list[str]]:
    """Print dataset summary information and return column classifications."""
    column_types = classify_columns(df)

    print("\n=== DATASET INFO ===")
    print(f"Shape: {df.shape}")
    print(f"Dataset size in memory: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print("\nMissing values:")
    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100
    summary = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct}).sort_values(
        ["missing_count", "missing_pct"], ascending=False
    )
    print(summary.to_string())
    print("\nDescriptive statistics:")
    print(df.describe(include="all").transpose().to_string())

    print("\nVariable classification:")
    for name, values in column_types.items():
        print(f"- {name.title():<11}: {len(values)}")
        if values:
            print(f"  {', '.join(values)}")

    logger.info("Dataset summary printed successfully.")
    return column_types


def prepare_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric copy of the dataframe for correlation and modeling."""
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    return numeric_df


def save_current_figure(output_path: Path) -> None:
    """Save the current figure with tight layout and close it."""
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_histograms(numeric_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot histograms for every numeric column."""
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_df.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, column in enumerate(numeric_df.columns):
        sns.histplot(numeric_df[column], kde=True, ax=axes[idx], color="#2a6fdb", bins=30)
        axes[idx].set_title(column)
        axes[idx].set_xlabel("")

    for idx in range(len(numeric_df.columns), len(axes)):
        axes[idx].axis("off")

    output_path = output_dir / FIGURE_FILENAMES["histograms"]
    save_current_figure(output_path)
    return output_path


def plot_boxplots(numeric_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot boxplots for every numeric column."""
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_df.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, column in enumerate(numeric_df.columns):
        sns.boxplot(x=numeric_df[column], ax=axes[idx], color="#ff8c42")
        axes[idx].set_title(column)
        axes[idx].set_xlabel("")

    for idx in range(len(numeric_df.columns), len(axes)):
        axes[idx].axis("off")

    output_path = output_dir / FIGURE_FILENAMES["boxplots"]
    save_current_figure(output_path)
    return output_path


def plot_correlation_heatmap(numeric_df: pd.DataFrame, output_dir: Path) -> tuple[Path, pd.DataFrame]:
    """Plot the full annotated correlation matrix."""
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap")
    output_path = output_dir / FIGURE_FILENAMES["correlation_heatmap"]
    save_current_figure(output_path)
    return output_path, corr


def plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot the target distribution with mean and median lines."""
    target = df[TARGET_COLUMN].dropna()
    mean_value = target.mean()
    median_value = target.median()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(target, kde=True, bins=30, ax=ax, color="#1f77b4")
    ax.axvline(mean_value, color="#d62728", linestyle="--", linewidth=2, label=f"Mean: {mean_value:.4f}")
    ax.axvline(median_value, color="#2ca02c", linestyle="-.", linewidth=2, label=f"Median: {median_value:.4f}")
    ax.set_title(f"{TARGET_COLUMN} Distribution")
    ax.set_xlabel(TARGET_COLUMN)
    ax.legend()

    output_path = output_dir / FIGURE_FILENAMES["target_distribution"]
    save_current_figure(output_path)
    return output_path


def get_top_target_correlations(df: pd.DataFrame, target_column: str) -> pd.Series:
    """Return correlations with the target sorted by absolute value."""
    numeric_df = df.select_dtypes(include=[np.number])
    if target_column not in numeric_df.columns:
        raise ValueError(f"Target column '{target_column}' must be numeric for correlation analysis.")

    correlations = numeric_df.corr()[target_column].drop(target_column)
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)


def plot_target_correlations(df: pd.DataFrame, output_dir: Path) -> tuple[Path, pd.Series]:
    """Plot correlations with the target as a horizontal bar chart."""
    correlations = get_top_target_correlations(df, TARGET_COLUMN)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.4 * len(correlations))))
    colors = np.where(correlations >= 0, "#2ca02c", "#d62728")
    ordered = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    ordered.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Correlation with {TARGET_COLUMN}")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Feature")
    ax.axvline(0, color="black", linewidth=1)

    output_path = output_dir / FIGURE_FILENAMES["target_correlations"]
    save_current_figure(output_path)
    return output_path, correlations


def plot_top_feature_scatterplots(df: pd.DataFrame, correlations: pd.Series, output_dir: Path) -> Path:
    """Plot scatterplots against the target for the six most correlated features."""
    top_features = correlations.abs().sort_values(ascending=False).head(6).index.tolist()
    n_cols = 2
    n_rows = int(np.ceil(len(top_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, feature in enumerate(top_features):
        sns.regplot(
            data=df,
            x=feature,
            y=TARGET_COLUMN,
            ax=axes[idx],
            scatter_kws={"alpha": 0.35, "s": 18},
            line_kws={"color": "crimson", "linewidth": 2},
        )
        axes[idx].set_title(f"{feature} vs {TARGET_COLUMN}")

    for idx in range(len(top_features), len(axes)):
        axes[idx].axis("off")

    output_path = output_dir / FIGURE_FILENAMES["scatter_top_features"]
    save_current_figure(output_path)
    return output_path


def plot_pairplot_top_features(df: pd.DataFrame, correlations: pd.Series, output_dir: Path) -> Path:
    """Plot a pairplot using the top five correlated features plus the target."""
    top_features = correlations.abs().sort_values(ascending=False).head(5).index.tolist()
    pairplot_cols = top_features + [TARGET_COLUMN]
    sample = df[pairplot_cols].dropna()
    if len(sample) > 1000:
        sample = sample.sample(n=1000, random_state=42)

    pair_grid = sns.pairplot(sample, corner=True, diag_kind="hist", plot_kws={"alpha": 0.4, "s": 18})
    pair_grid.fig.suptitle("Pairplot of Top Correlated Features", y=1.02)

    output_path = output_dir / FIGURE_FILENAMES["pairplot_top_features"]
    pair_grid.fig.tight_layout()
    pair_grid.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(pair_grid.fig)
    return output_path


def prepare_modeling_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for machine learning."""
    feature_df = df.drop(columns=[TARGET_COLUMN]).select_dtypes(include=[np.number]).copy()
    target = df[TARGET_COLUMN].copy()

    imputer = SimpleImputer(strategy="median")
    feature_df.loc[:, :] = imputer.fit_transform(feature_df)
    return feature_df, target


def train_rf_model(df: pd.DataFrame) -> tuple[RandomForestRegressor, pd.DataFrame, pd.Series, pd.Series, pd.Series, float, float]:
    """Train a random forest regressor and return predictions and metrics."""
    sample = df.sample(n=min(len(df), 20000), random_state=42) if len(df) > 20000 else df
    x, y = prepare_modeling_data(sample)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_samples=0.6,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    return model, x_test, y_test, pd.Series(y_pred, index=y_test.index), x.columns.to_series(), rmse, r2


def plot_rf_feature_importance(model: RandomForestRegressor, feature_names: pd.Index, output_dir: Path) -> pd.Series:
    """Plot feature importance from a fitted random forest."""
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(importances))))
    importances.plot(kind="barh", ax=ax, color="#6a5acd")
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    output_path = output_dir / FIGURE_FILENAMES["rf_feature_importance"]
    save_current_figure(output_path)
    return importances


def plot_actual_vs_predicted(y_test: pd.Series, y_pred: pd.Series, output_dir: Path) -> Path:
    """Plot actual versus predicted target values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.5, s=22, color="#1f77b4")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", linewidth=2)
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    output_path = output_dir / FIGURE_FILENAMES["actual_vs_predicted"]
    save_current_figure(output_path)
    return output_path


def plot_residual_distribution(y_test: pd.Series, y_pred: pd.Series, output_dir: Path) -> Path:
    """Plot the residual distribution."""
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, ax=ax, color="#ff7f0e")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual")

    output_path = output_dir / FIGURE_FILENAMES["residual_distribution"]
    save_current_figure(output_path)
    return output_path


def create_feature_summary(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create a feature summary CSV for numerical features."""
    numeric_df = df.select_dtypes(include=[np.number])
    summary = pd.DataFrame(
        {
            "mean": numeric_df.mean(),
            "median": numeric_df.median(),
            "std": numeric_df.std(),
            "min": numeric_df.min(),
            "max": numeric_df.max(),
            "skewness": numeric_df.skew(),
        }
    )
    output_path = output_dir / "feature_summary.csv"
    summary.to_csv(output_path, index_label="feature")
    return output_path


def print_top_correlations(correlations: pd.Series) -> None:
    """Print top correlations with the target."""
    print("\n=== TOP CORRELATIONS WITH TARGET ===")
    top = correlations.abs().sort_values(ascending=False)
    display = pd.DataFrame(
        {
            "correlation": correlations.loc[top.index],
            "abs_correlation": top,
        }
    )
    print(display.to_string())


def print_feature_importance(importances: pd.Series) -> None:
    """Print random forest feature importance."""
    print("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")
    print(importances.sort_values(ascending=False).to_string())


def print_generated_files(generated_files: list[Path], project_root: Path) -> None:
    """Print the list of generated files relative to the project root."""
    print("\n=== GENERATED FILES ===")
    for path in generated_files:
        print(f"- {path.relative_to(project_root)}")


def run_eda() -> dict[str, Any]:
    """Run the full Track 1 EDA workflow."""
    logger = configure_logging()
    setup_plot_style()

    project_root = get_project_root()
    dataset_path = get_dataset_path(project_root)
    output_dir = get_output_dir(project_root)

    df = load_dataset(dataset_path)
    column_types = print_dataset_summary(df, logger)

    numeric_df = prepare_numeric_data(df)
    if TARGET_COLUMN not in numeric_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is not numeric after loading.")

    generated_files: list[Path] = []
    generated_files.append(plot_histograms(numeric_df, output_dir))
    generated_files.append(plot_boxplots(numeric_df, output_dir))
    generated_files.append(plot_correlation_heatmap(numeric_df, output_dir)[0])
    generated_files.append(plot_target_distribution(df, output_dir))

    target_correlations = get_top_target_correlations(df, TARGET_COLUMN)
    generated_files.append(plot_target_correlations(df, output_dir)[0])
    generated_files.append(plot_top_feature_scatterplots(df, target_correlations, output_dir))
    generated_files.append(plot_pairplot_top_features(df, target_correlations, output_dir))

    model, x_test, y_test, y_pred, feature_names, rmse, r2 = train_rf_model(df)
    importances = plot_rf_feature_importance(model, feature_names, output_dir)
    generated_files.append(output_dir / FIGURE_FILENAMES["rf_feature_importance"])
    generated_files.append(plot_actual_vs_predicted(y_test, y_pred, output_dir))
    generated_files.append(plot_residual_distribution(y_test, y_pred, output_dir))

    generated_files.append(create_feature_summary(df, output_dir))

    print_top_correlations(target_correlations)
    print_feature_importance(importances)

    print(f"\nModel performance: RMSE={rmse:.6f}, R2={r2:.6f}")
    print_generated_files(generated_files, project_root)

    return {
        "dataset": df,
        "column_types": column_types,
        "target_correlations": target_correlations,
        "feature_importance": importances,
        "generated_files": generated_files,
        "rmse": rmse,
        "r2": r2,
    }


def main() -> None:
    """Entry point for script execution."""
    try:
        run_eda()
    except Exception as exc:  # pragma: no cover
        logging.exception("Track 1 EDA failed: %s", exc)
        raise


if __name__ == "__main__":
    main()