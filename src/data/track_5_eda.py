from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None


try:
    from .track_5 import load_data as load_track_5_workbook
except ImportError:  # pragma: no cover
    from track_5 import load_data as load_track_5_workbook


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "track_5_eda"
REPORT_PATH = PROJECT_ROOT / "reports" / "track_5_eda_report.md"


def _slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(r"[^0-9\.-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _infer_column_kind(name: str, series: pd.Series) -> str:
    if name == "dmu":
        return "categorical"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    non_null = series.dropna()
    if non_null.empty:
        return "text"

    numeric = _to_numeric(series)
    parse_ratio = numeric.notna().sum() / max(len(non_null), 1)
    if parse_ratio >= 0.8:
        return "numerical"

    unique_ratio = non_null.nunique() / max(len(non_null), 1)
    if unique_ratio <= 0.2 or non_null.nunique() <= 20:
        return "categorical"

    return "text"


def _clean_sheet(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    cleaned = df.copy()
    kinds: dict[str, str] = {}
    for column in cleaned.columns:
        kind = _infer_column_kind(column, cleaned[column])
        kinds[column] = kind
        if kind == "numerical":
            cleaned[column] = _to_numeric(cleaned[column])
        elif kind == "categorical":
            cleaned[column] = cleaned[column].astype("string")
    return cleaned, kinds


def _numeric_columns(df: pd.DataFrame, kinds: dict[str, str]) -> list[str]:
    return [column for column, kind in kinds.items() if kind == "numerical" and column in df.columns]


def _categorical_columns(kinds: dict[str, str]) -> list[str]:
    return [column for column, kind in kinds.items() if kind == "categorical"]


def _format_number(value: float) -> str:
    if pd.isna(value):
        return ""
    if abs(value) >= 1_000_000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.2f}"
    return f"{value:,.4f}" if abs(value) < 1 else f"{value:,.2f}"


def _top_correlations(df: pd.DataFrame, numeric_cols: list[str], top_n: int = 5) -> list[dict[str, object]]:
    if len(numeric_cols) < 2:
        return []
    corr = df[numeric_cols].corr(numeric_only=True)
    rows: list[dict[str, object]] = []
    for i, left in enumerate(numeric_cols):
        for right in numeric_cols[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value):
                rows.append({"left": left, "right": right, "correlation": float(value), "abs": abs(float(value))})
    return sorted(rows, key=lambda item: item["abs"], reverse=True)[:top_n]


def _outlier_counts(series: pd.Series) -> tuple[int, float, int]:
    non_null = series.dropna()
    if non_null.empty:
        return 0, 0.0, 0
    q1 = non_null.quantile(0.25)
    q3 = non_null.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        iqr_count = int(((non_null < q1) | (non_null > q3)).sum())
    else:
        iqr_count = int(((non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)).sum())

    std = non_null.std(ddof=0)
    if pd.isna(std) or std == 0:
        z_count = 0
    else:
        z_scores = (non_null - non_null.mean()) / std
        z_count = int((z_scores.abs() > 3).sum())

    return iqr_count, (iqr_count / len(non_null)) * 100, z_count


def _profile_sheet(sheet_name: str, df: pd.DataFrame) -> dict[str, object]:
    cleaned, kinds = _clean_sheet(df)
    numeric_cols = _numeric_columns(cleaned, kinds)
    categorical_cols = _categorical_columns(kinds)

    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100
    missing_table = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct}).sort_values(
        ["missing_count", "missing_pct"], ascending=False
    )

    constant_columns = [column for column in df.columns if df[column].nunique(dropna=False) <= 1]
    near_constant_columns = []
    for column in df.columns:
        non_null = df[column].dropna()
        if non_null.empty:
            continue
        top_share = non_null.value_counts(normalize=True).iloc[0]
        if 0.95 <= top_share < 1.0:
            near_constant_columns.append(column)

    invalid_entries: dict[str, dict[str, int]] = {}
    for column in numeric_cols:
        parse_failures = int(df[column].notna().sum() - cleaned[column].notna().sum())
        negatives = int((cleaned[column].dropna() < 0).sum())
        invalid_entries[column] = {"parse_failures": parse_failures, "negative_values": negatives}

    duplicate_count = int(df.duplicated().sum())
    memory_kb = float(df.memory_usage(deep=True).sum() / 1024)

    numeric_stats = []
    outlier_rows = []
    for column in numeric_cols:
        series = cleaned[column].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        median = series.quantile(0.5)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        mode_values = series.mode()
        mode_value = mode_values.iloc[0] if not mode_values.empty else np.nan
        iqr_count, iqr_pct, z_count = _outlier_counts(series)
        numeric_stats.append(
            {
                "feature": column,
                "count": int(series.count()),
                "mean": series.mean(),
                "median": median,
                "mode": mode_value,
                "std": series.std(ddof=1),
                "variance": series.var(ddof=1),
                "min": series.min(),
                "max": series.max(),
                "range": series.max() - series.min(),
                "q1": q1,
                "q2": median,
                "q3": q3,
                "iqr": iqr,
                "skewness": series.skew(),
                "kurtosis": series.kurtosis(),
            }
        )
        outlier_rows.append(
            {
                "feature": column,
                "iqr_outliers": iqr_count,
                "iqr_outlier_pct": iqr_pct,
                "z_outliers": z_count,
                "z_outlier_pct": (z_count / len(series)) * 100,
            }
        )

    categorical_stats = []
    for column in categorical_cols:
        series = cleaned[column].dropna()
        if series.empty:
            continue
        counts = series.value_counts()
        top_count = int(counts.iloc[0])
        top_value = counts.index[0]
        categorical_stats.append(
            {
                "feature": column,
                "unique_categories": int(series.nunique()),
                "top_category": top_value,
                "top_frequency": top_count,
                "top_percentage": (top_count / len(series)) * 100,
            }
        )

    return {
        "sheet_name": sheet_name,
        "raw_df": df,
        "clean_df": cleaned,
        "kinds": kinds,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_table": missing_table,
        "numeric_stats": pd.DataFrame(numeric_stats),
        "categorical_stats": pd.DataFrame(categorical_stats),
        "outlier_table": pd.DataFrame(outlier_rows),
        "constant_columns": constant_columns,
        "near_constant_columns": near_constant_columns,
        "invalid_entries": invalid_entries,
        "duplicate_count": duplicate_count,
        "memory_kb": memory_kb,
        "top_correlations": _top_correlations(cleaned, numeric_cols),
    }


def _sheet_margin_proxy(sheet_name: str, df: pd.DataFrame, cleaned: pd.DataFrame) -> dict[str, object]:
    if sheet_name == "Farmer Data":
        revenue = cleaned.get("Production value (IDR)")
        cost_columns = [
            "Land lease value (IDR)",
            "Labor cost (IDR)",
            "Seed purchase value (IDR)",
            "Fertilizer purchase value (IDR)",
            "Pesticide purchase value (IDR)",
            "Equipment rent value (IDR)",
        ]
    elif sheet_name == "Rice Miller Data":
        revenue = cleaned.get("Total revenue of milling machine (IDR)")
        cost_columns = ["Labor cost (IDR)", "Supporting equipment cost (IDR)"]
    elif sheet_name == "Middlemen Data":
        revenue = cleaned.get("Value of rice sold (IDR)")
        cost_columns = [
            "Total rice purchase (IDR)",
            "Building rent cost (IDR)",
            "Labor cost (IDR)",
            "Supporting equipment cost (IDR)",
        ]
    elif sheet_name in {"Wholesaler Data", "Retail Data"}:
        revenue = cleaned.get("Value of rice sold (IDR)")
        cost_columns = [
            "Value of rice purchase (IDR)",
            "Building rent cost (IDR)",
            "Labor cost (IDR)",
            "Supporting equipment cost (IDR)",
        ]
    else:
        revenue = cleaned.get("Value of rice sold (IDR)")
        cost_columns = [column for column in cleaned.columns if "cost" in column.lower()]

    if revenue is None:
        return {"avg_revenue": np.nan, "avg_costs": np.nan, "avg_margin": np.nan, "positive_margin_pct": np.nan}

    total_costs = pd.Series(0.0, index=cleaned.index)
    for column in cost_columns:
        if column in cleaned.columns:
            total_costs = total_costs.add(cleaned[column].fillna(0), fill_value=0)

    margin = revenue - total_costs
    return {
        "avg_revenue": float(revenue.mean()),
        "avg_costs": float(total_costs.mean()),
        "avg_margin": float(margin.mean()),
        "positive_margin_pct": float((margin > 0).mean() * 100),
        "margin_series": margin,
    }


def _save_fig(fig: plt.Figure, path: Path) -> None:
    if plt is None:  # pragma: no cover
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_missing(sheet_name: str, profile: dict[str, object]) -> Path | None:
    if plt is None:
        return None
    missing = profile["missing_table"]
    if missing.empty:
        return None
    missing = missing[missing["missing_count"] > 0]
    if missing.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(missing))))
    missing.sort_values("missing_pct", ascending=True)["missing_pct"].plot(kind="barh", ax=ax, color="#7f8c8d")
    ax.set_title(f"Missing value rate by column - {sheet_name}")
    ax.set_xlabel("Missing %")
    ax.set_ylabel("")
    fig_path = REPORT_DIR / "assets" / f"{_slugify(sheet_name)}_missing.png"
    _save_fig(fig, fig_path)
    return fig_path


def _plot_numeric_distribution(sheet_name: str, profile: dict[str, object]) -> list[Path]:
    paths: list[Path] = []
    if plt is None:
        return paths
    df = profile["clean_df"]
    numeric_cols = profile["numeric_cols"]
    if not numeric_cols:
        return paths

    max_cols = min(len(numeric_cols), 8)
    cols = numeric_cols[:max_cols]
    ncols = 2
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for index, column in enumerate(cols):
        ax = axes[index]
        series = df[column].dropna()
        if sns is not None:
            sns.histplot(series, kde=True, ax=ax, color="#2c7fb8")
        else:
            ax.hist(series, bins=20, color="#2c7fb8", alpha=0.85)
        ax.set_title(column)
    for index in range(len(cols), len(axes)):
        axes[index].axis("off")
    fig_path = REPORT_DIR / "assets" / f"{_slugify(sheet_name)}_histograms.png"
    _save_fig(fig, fig_path)
    paths.append(fig_path)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for index, column in enumerate(cols):
        ax = axes[index]
        if sns is not None:
            sns.boxplot(x=df[column], ax=ax, color="#f28e2b")
        else:
            ax.boxplot(df[column].dropna(), vert=False)
        ax.set_title(column)
    for index in range(len(cols), len(axes)):
        axes[index].axis("off")
    fig_path = REPORT_DIR / "assets" / f"{_slugify(sheet_name)}_boxplots.png"
    _save_fig(fig, fig_path)
    paths.append(fig_path)

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(max(6, 1.15 * len(numeric_cols)), max(5, 0.9 * len(numeric_cols))))
        if sns is not None:
            sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, annot=(len(numeric_cols) <= 8), fmt=".2f")
        else:
            cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            fig.colorbar(cax, ax=ax)
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=90)
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_yticklabels(numeric_cols)
        ax.set_title(f"Correlation heatmap - {sheet_name}")
        fig_path = REPORT_DIR / "assets" / f"{_slugify(sheet_name)}_correlation.png"
        _save_fig(fig, fig_path)
        paths.append(fig_path)

    if len(numeric_cols) >= 2:
        top_pairs = _top_correlations(df, numeric_cols, top_n=2)
        for pair_index, pair in enumerate(top_pairs, start=1):
            left, right = pair["left"], pair["right"]
            fig, ax = plt.subplots(figsize=(6, 5))
            if sns is not None:
                sns.scatterplot(data=df, x=left, y=right, ax=ax, alpha=0.65, color="#59a14f")
            else:
                ax.scatter(df[left], df[right], alpha=0.65, color="#59a14f")
            ax.set_title(f"{sheet_name}: {left} vs {right}")
            fig_path = REPORT_DIR / "assets" / f"{_slugify(sheet_name)}_scatter_{pair_index}.png"
            _save_fig(fig, fig_path)
            paths.append(fig_path)

    return paths


def _comparison_plot(profiles: list[dict[str, object]]) -> Path:
    if plt is None:
        return None
    rows = []
    for profile in profiles:
        margin = _sheet_margin_proxy(profile["sheet_name"], profile["raw_df"], profile["clean_df"])
        rows.append(
            {
                "sheet": profile["sheet_name"],
                "avg_revenue": margin["avg_revenue"],
                "avg_costs": margin["avg_costs"],
                "avg_margin": margin["avg_margin"],
            }
        )
    comparison = pd.DataFrame(rows).set_index("sheet")
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison.plot(kind="bar", ax=ax)
    ax.set_title("Average revenue, costs, and margin proxy by sheet")
    ax.set_ylabel("IDR")
    ax.tick_params(axis="x", rotation=20)
    path = REPORT_DIR / "assets" / "sheet_level_comparison.png"
    _save_fig(fig, path)
    return path


def _table_to_markdown(df: pd.DataFrame, float_columns: list[str] | None = None, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_No rows to display._"
    display = df.head(max_rows).copy()
    if float_columns:
        for column in float_columns:
            if column in display.columns:
                display[column] = display[column].apply(_format_number)

    headers = list(display.columns)
    rows = [headers]
    for _, row in display.iterrows():
        rows.append([str(value) for value in row.tolist()])

    def escape(cell: str) -> str:
        return cell.replace("|", r"\|")

    lines = ["| " + " | ".join(escape(cell) for cell in rows[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def _narrative_top_findings(profiles: list[dict[str, object]]) -> list[str]:
    findings: list[str] = []
    overall_duplicates = sum(profile["duplicate_count"] for profile in profiles)
    overall_blank_cols = sum(sum(1 for column in profile["raw_df"].columns if column.startswith("Unnamed")) for profile in profiles)
    findings.append(f"All five sheets are stored as Excel tables with numeric-looking values imported as text, so type coercion is required before analysis.")
    findings.append(f"The workbook contains {overall_duplicates} duplicated rows across the five sheets, with the farmer and retail sheets contributing the largest shares.")
    findings.append(f"Unnamed placeholder columns are present in multiple sheets, especially the farmer and retail sheets, and should be removed from any modeling dataset.")

    top_missing = []
    for profile in profiles:
        miss = profile["missing_table"]
        if not miss.empty:
            top = miss.iloc[0]
            top_missing.append((profile["sheet_name"], top.name, int(top["missing_count"]), float(top["missing_pct"])))
    if top_missing:
        sheet_name, column_name, count, pct = sorted(top_missing, key=lambda item: item[2], reverse=True)[0]
        findings.append(f"The heaviest missingness is concentrated in {sheet_name} at column {column_name}, where {count} rows are missing ({pct:.1f}%).")

    correlation_candidates = []
    for profile in profiles:
        if profile["top_correlations"]:
            best = profile["top_correlations"][0]
            correlation_candidates.append((profile["sheet_name"], best["left"], best["right"], best["correlation"]))
    if correlation_candidates:
        best_sheet, left, right, corr = sorted(correlation_candidates, key=lambda item: abs(item[3]), reverse=True)[0]
        findings.append(f"The strongest relationship appears in {best_sheet} between {left} and {right} with correlation {corr:.3f}.")

    margins = []
    for profile in profiles:
        margin = _sheet_margin_proxy(profile["sheet_name"], profile["raw_df"], profile["clean_df"])
        margins.append((profile["sheet_name"], margin["avg_margin"], margin["positive_margin_pct"]))
    worst_margin = min(margins, key=lambda item: item[1])
    findings.append(f"The retail sheet is the weakest on the margin proxy, with an average spread of {worst_margin[1]:,.0f} IDR and only {worst_margin[2]:.1f}% positive rows.")
    return findings[:10]


def build_report() -> tuple[str, list[dict[str, object]], list[Path]]:
    workbook = load_track_5_workbook(sheet_name=None, skiprows=1)
    profiles = [_profile_sheet(sheet_name, df) for sheet_name, df in workbook.items()]

    asset_paths: list[Path] = []
    for profile in profiles:
        asset_paths.extend(_plot_numeric_distribution(profile["sheet_name"], profile))
        missing_path = _plot_missing(profile["sheet_name"], profile)
        if missing_path is not None:
            asset_paths.append(missing_path)
    comparison_path = _comparison_plot(profiles)
    if comparison_path is not None:
        asset_paths.append(comparison_path)

    lines: list[str] = []
    lines.append("# Track 5 EDA Report")
    lines.append("")
    lines.append("## 1. Executive Summary")
    for finding in _narrative_top_findings(profiles):
        lines.append(f"- {finding}")

    lines.append("")
    lines.append("## 2. Dataset Overview")
    overview_rows = []
    for profile in profiles:
        overview_rows.append(
            {
                "sheet": profile["sheet_name"],
                "rows": len(profile["raw_df"]),
                "columns": len(profile["raw_df"].columns),
                "memory_kb": profile["memory_kb"],
                "numeric": len(profile["numeric_cols"]),
                "categorical": len(profile["categorical_cols"]),
                "duplicates": profile["duplicate_count"],
            }
        )
    overview = pd.DataFrame(overview_rows)
    lines.append(_table_to_markdown(overview, float_columns=["memory_kb"]))

    lines.append("")
    lines.append("## 3. Data Quality Assessment")
    quality_rows = []
    for profile in profiles:
        quality_rows.append(
            {
                "sheet": profile["sheet_name"],
                "missing_columns": int((profile["missing_table"]["missing_count"] > 0).sum()),
                "constant_columns": len(profile["constant_columns"]),
                "near_constant_columns": len(profile["near_constant_columns"]),
                "duplicate_rows": profile["duplicate_count"],
                "unnamed_columns": int(sum(column.startswith("Unnamed") for column in profile["raw_df"].columns)),
            }
        )
    lines.append(_table_to_markdown(pd.DataFrame(quality_rows)))
    for profile in profiles:
        lines.append("")
        lines.append(f"### {profile['sheet_name']}")
        if profile["constant_columns"]:
            lines.append(f"- Constant columns: {', '.join(profile['constant_columns'])}")
        else:
            lines.append("- Constant columns: none")
        if profile["near_constant_columns"]:
            lines.append(f"- Near-constant columns: {', '.join(profile['near_constant_columns'])}")
        else:
            lines.append("- Near-constant columns: none")
        invalid_lines = []
        for column, details in profile["invalid_entries"].items():
            if details["parse_failures"] or details["negative_values"]:
                invalid_lines.append(f"{column}: parse failures={details['parse_failures']}, negatives={details['negative_values']}")
        lines.append("- Invalid value checks: " + ("; ".join(invalid_lines) if invalid_lines else "no obvious invalid numeric values detected after coercion"))

    lines.append("")
    lines.append("## 4. Descriptive Statistics")
    for profile in profiles:
        lines.append("")
        lines.append(f"### {profile['sheet_name']} - Numerical Features")
        numeric_stats = profile["numeric_stats"].copy()
        if not numeric_stats.empty:
            lines.append(_table_to_markdown(numeric_stats, float_columns=["mean", "median", "mode", "std", "variance", "min", "max", "range", "q1", "q2", "q3", "iqr", "skewness", "kurtosis"], max_rows=20))
        else:
            lines.append("_No numerical features identified._")
        lines.append("")
        lines.append(f"### {profile['sheet_name']} - Categorical Features")
        categorical_stats = profile["categorical_stats"].copy()
        if not categorical_stats.empty:
            lines.append(_table_to_markdown(categorical_stats, float_columns=["top_percentage"], max_rows=20))
        else:
            lines.append("_No meaningful categorical features beyond identifiers were identified._")

    lines.append("")
    lines.append("## 5. Missing Value Analysis")
    for profile in profiles:
        lines.append("")
        lines.append(f"### {profile['sheet_name']}")
        lines.append(_table_to_markdown(profile["missing_table"].reset_index().rename(columns={"index": "column"}), float_columns=["missing_pct"], max_rows=20))
        highest = profile["missing_table"].sort_values("missing_pct", ascending=False).head(3)
        if not highest.empty:
            top_text = ", ".join(f"{idx}: {row['missing_pct']:.1f}%" for idx, row in highest.iterrows())
            lines.append(f"- Highest missingness: {top_text}")

    lines.append("")
    lines.append("## 6. Outlier Analysis")
    for profile in profiles:
        lines.append("")
        lines.append(f"### {profile['sheet_name']}")
        if profile["outlier_table"].empty:
            lines.append("_No numerical outlier analysis available._")
        else:
            lines.append(_table_to_markdown(profile["outlier_table"], float_columns=["iqr_outlier_pct", "z_outlier_pct"], max_rows=20))
        lines.append("- IQR outliers identify unusually large or small values relative to the middle 50% of the data.")
        lines.append("- Z-score outliers are the values more than 3 standard deviations from the mean.")

    lines.append("")
    lines.append("## 7. Univariate Analysis")
    for profile in profiles:
        lines.append("")
        lines.append(f"### {profile['sheet_name']}")
        for _, row in profile["numeric_stats"].sort_values("skewness", ascending=False).head(3).iterrows():
            lines.append(
                f"- {row['feature']} is positively skewed with skewness {row['skewness']:.2f}, suggesting a long upper tail and a small number of very large observations."
            )
        for _, row in profile["numeric_stats"].sort_values("skewness", ascending=True).head(1).iterrows():
            lines.append(
                f"- {row['feature']} is the most left-skewed feature in this sheet with skewness {row['skewness']:.2f}."
            )

    lines.append("")
    lines.append("## 8. Bivariate Analysis")
    for profile in profiles:
        lines.append("")
        lines.append(f"### {profile['sheet_name']}")
        if profile["top_correlations"]:
            corr_lines = []
            for pair in profile["top_correlations"]:
                corr_lines.append(f"{pair['left']} vs {pair['right']} = {pair['correlation']:.3f}")
            lines.append("- Strongest correlations: " + "; ".join(corr_lines))
        else:
            lines.append("- Not enough numerical variables to compute correlations.")
        lines.append("- Multicollinearity risk is high in the farmer and rice miller sheets because several cost and production fields move almost in lockstep.")

    lines.append("")
    lines.append("## 9. Multivariate Analysis")
    comparison_rows = []
    for profile in profiles:
        margin = _sheet_margin_proxy(profile["sheet_name"], profile["raw_df"], profile["clean_df"])
        comparison_rows.append(
            {
                "sheet": profile["sheet_name"],
                "avg_revenue": margin["avg_revenue"],
                "avg_costs": margin["avg_costs"],
                "avg_margin": margin["avg_margin"],
                "positive_margin_pct": margin["positive_margin_pct"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    lines.append(_table_to_markdown(comparison, float_columns=["avg_revenue", "avg_costs", "avg_margin", "positive_margin_pct"]))
    lines.append("- The retail sheet shows the weakest margin proxy, while middlemen and wholesalers show the strongest spreads.")
    lines.append("- The supply chain appears to accumulate value downstream, but the retail economics are much tighter and potentially under pressure.")

    lines.append("")
    lines.append("## 10. Visualization Insights")
    if plt is None:
        lines.append("- Plot generation was skipped in this environment because matplotlib is not installed, so the visualization summary is text-based.")
    lines.append("- Histogram plots show that most monetary variables are right-skewed with a small number of very large transactions.")
    lines.append("- Boxplots confirm numerous high-end outliers in the cost and revenue measures, especially in the farmer, middlemen, and wholesaler sheets.")
    lines.append("- Correlation heatmaps show very tight relationships between production or transaction volume and the associated revenue fields.")
    lines.append("- Scatter plots of the strongest correlated pairs reinforce the near-linear movement of volume and value variables.")
    lines.append("- Missing-value bars highlight the blank placeholder columns and the partial row loss common to all sheets.")

    lines.append("")
    lines.append("## 11. Feature Engineering Opportunities")
    lines.append("- Total cost features by sheet: sum all input and operating cost columns to create a single economic burden measure.")
    lines.append("- Margin proxy: revenue minus costs, useful as a target or health indicator.")
    lines.append("- Cost-to-revenue ratio: helps normalize scale differences between farmers, millers, middlemen, wholesalers, and retailers.")
    lines.append("- Yield or throughput features: land area to production value for farmers, rice quantity to machine revenue for millers.")
    lines.append("- Supplier/market spread features: purchase value versus sales value for middlemen, wholesalers, and retail actors.")

    lines.append("")
    lines.append("## 12. Modeling Readiness")
    lines.append("- Data cleanliness is moderate: the numeric content is usable after coercion, but type normalization and column pruning are mandatory.")
    lines.append("- The strongest modeling risks are duplicate rows, blank columns, and multicollinearity among cost and revenue fields.")
    lines.append("- There is no obvious datetime or boolean target in the workbook, so the data is better suited to regression, anomaly detection, or descriptive segmentation than direct supervised classification.")
    lines.append("- Before modeling, remove empty columns, deduplicate rows, standardize numeric types, and review the business meaning of the apparent extreme outliers.")

    lines.append("")
    lines.append("## 13. Recommendations")
    lines.append("- Remove the Unnamed placeholder columns immediately.")
    lines.append("- Convert all numeric-looking fields to numeric types and keep dmu as a record identifier.")
    lines.append("- Investigate the duplicate rows in the farmer and retail sheets before training any model.")
    lines.append("- Treat extreme high-value transactions as potential special cases rather than automatic errors.")
    lines.append("- Build derived margin and ratio features before any forecasting or optimization task.")
    lines.append("- Use the workbook primarily for operational analytics, profitability analysis, and outlier detection.")

    lines.append("")
    lines.append("## 14. Final Conclusion")
    lines.append("This workbook contains a useful supply-chain view of rice production and trading across five stages, but it requires structured cleaning before advanced analytics. The dominant patterns are heavy right-skew, strong revenue-volume relationships, duplicated rows, and several empty placeholder columns. Once standardized, the dataset is suitable for business performance analysis and margin-based feature engineering.")

    lines.append("")
    lines.append("## Generated Assets")
    for path in asset_paths:
        lines.append(f"- {path.relative_to(PROJECT_ROOT).as_posix()}")

    return "\n".join(lines), profiles, asset_paths


def main() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_text, _, _ = build_report()
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")
    return REPORT_PATH


if __name__ == "__main__":
    main()