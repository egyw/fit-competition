"""
Track 4 – EDA: AI-Based Vector-Borne Disease Dataset
=====================================================
Dataset: 300 patients, 109 columns (symptoms, vitals, lab results, diagnoses)
Diseases: Malaria, Dengue, Yellow Fever, Typhoid, Others (multi-label)

Run:
    python track_4.py

Outputs are saved to ./eda_outputs/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = "data/raw/Track 4/Tabular dataset for AI-based vector-borne disease/Tabular dataset for AI-based vector-borne disease/data.csv"
OUTPUT_DIR = Path("eda_outputs/track_4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = "Set2"
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)

# Disease target columns (binary 0/1)
DISEASE_COLS = {
    "Malaria":       "Maladies diagnostiquées/Paludisme (Malaria)",
    "Dengue":        "Maladies diagnostiquées/Dengue",
    "Yellow Fever":  "Maladies diagnostiquées/Fièvre jaune (yellow fever)",
    "Typhoid":       "Maladies diagnostiquées/Fièvre Typhoïde (Thyphoid fever)",
    "Others":        "Maladies diagnostiquées/Autres maladies diagnostiqué (Others diseases)",
}

# Symptom columns (OUI/NON)
SYMPTOM_COLS = {
    "Fever (48h)":            "Fièvre depuis 48 heures(Fever 48 hrs)",
    "Fever (7 days)":         "Fièvre au cours des 7 derniers jours (Fever in the last 7 days)",
    "Headache":               "Maux de tête (headache)",
    "Nausea":                 "Nausée (Nausea)",
    "Vomiting":               "Vomissement (Vomiting)",
    "Joint Pain":             "Douleur articulaire (Joint pain)",
    "Muscle Pain":            "Douleur musculaire ( Muscle pain)",
    "Skin Rash":              "Éruptions cutanées ou Exanthème (Skin rash or Exanthema)",
    "Bleeding":               "Saignement/ Manifestations hémorragiques (Bleeding)",
    "Abdominal Pain":         "Douleur abdominale (stomac pain)",
    "Diarrhea":               "Diarrhée  (Diarrhea)",
    "Cough":                  "Toux (Cough)",
    "Prostration":            "Prostration",
    "Consciousness Trouble":  "Troubles de la conscience (Consciousness trouble)",
    "Shiver/Cold":            "Sensation de frissons ou de froid (Shiver or cold sensation)",
    "Drowsiness/Lethargy":    "Somnolence ou léthargie  (Drowsiness or lethargy)",
    "Retro-orbital Pain":     " Douleur rétro-orbitrale (Retro-orbital pain)",
    "Hepatomegaly":           "Hépatomégalie (Hepatomegaly)",
    "Icterus":                "Ictère (Icterus)",
    "Respiratory Distress":   "Détresse respiratoire (Respiratory distress)",
    "Anorexia":               "Perte d'appétit ou Anorexie (Loss of appetite or Anorexia)",
    "Dizziness":              "Vertige (Dizzy)",
}

# Numeric / lab columns
NUMERIC_COLS = {
    "Age":             "Âge (Age)",
    "Weight (kg)":     "Poids (Weight)",
    "Temperature (°C)":"Température axillaire (médiane IQR) (°C) /Axillary temperature (median IQR) (°C)",
    "Heart Rate":      "Fréquence du pouls (battements/m in ± SD)./ Pulse rate (mean beats/min ± SD) - Shock ou Myocarditis",
    "Resp. Rate":      "Fréquence respiratoire (médiane IQR) / Respiratory rate (median breaths/min IQR)",
    "Hematocrit":      "Hematocrite (Hematrocrit)",
    "Platelets":       "Numération plaquettaire Platelet count",
    "WBC":             "Nombre de globules blancs (cellules/ML) / White blood cell count / WBC count (cells/ML)",
    "Neutrophils":     "Neutrophiles / Neutrophils",
    "Lymphocytes":     "Lymphocytes ",
    "Transaminases":   "Transaminases (Transaminases)",
    "Creatinine":      "Créatinine élevée / Elevated Creatinine",
}

# Comorbidity columns (OUI/NON)
COMORBIDITY_COLS = {
    "Diabetes":        "Diabète ",
    "Hypertension":    "Hypertension artérielle",
    "Allergies":       "Allergies",
    "Cancer":          "Cancer",
    "Asthma":          "Asthme (Astma)",
    "Household Dengue":"Dengue dans le menage (Household Dengue)",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def oui_non_to_binary(series: pd.Series) -> pd.Series:
    """Convert OUI/NON strings to 1/0 integers."""
    return series.str.strip().str.upper().map({"OUI": 1, "NON": 0})


def save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Load & Basic Clean ────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")

    # Convert Age to numeric (some entries are floats like '2.8')
    df["Âge (Age)"] = pd.to_numeric(df["Âge (Age)"], errors="coerce")

    # Numeric columns stored as strings with commas → floats
    for col in df.columns:
        if df[col].dtype == object:
            converted = df[col].str.replace(",", ".", regex=False)
            numeric = pd.to_numeric(converted, errors="coerce")
            if numeric.notna().sum() > 0.3 * df[col].notna().sum():
                df[col] = numeric

    return df


# ── 1. Dataset Overview ───────────────────────────────────────────────────────

def overview(df: pd.DataFrame) -> None:
    section("1. Dataset Overview")
    print(f"  Shape            : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Numeric columns  : {df.select_dtypes(include='number').shape[1]}")
    print(f"  Categorical cols : {df.select_dtypes(include='object').shape[1]}")
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    print(f"  Total missing    : {total_missing} / {total_cells} ({100*total_missing/total_cells:.1f}%)")

    # Gender distribution
    gender = df["Genre (Gender)"].str.strip().value_counts()
    print(f"\n  Gender distribution:\n{gender.to_string()}")


# ── 2. Missing Values Heatmap ─────────────────────────────────────────────────

def plot_missing(df: pd.DataFrame) -> None:
    section("2. Missing Values")

    missing_pct = df.isnull().mean() * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

    print(f"  Columns with any missing: {len(missing_pct)}")
    print(f"  Top 10 by % missing:\n{missing_pct.head(10).round(1).to_string()}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#e74c3c" if v > 50 else "#f39c12" if v > 20 else "#3498db"
              for v in missing_pct.values]
    ax.bar(range(len(missing_pct)), missing_pct.values, color=colors)
    ax.set_xticks(range(len(missing_pct)))
    ax.set_xticklabels(
        [c[:40] + "…" if len(c) > 40 else c for c in missing_pct.index],
        rotation=90, fontsize=7
    )
    ax.axhline(50, color="red", linestyle="--", linewidth=1, label=">50% missing")
    ax.axhline(20, color="orange", linestyle="--", linewidth=1, label=">20% missing")
    ax.set_ylabel("Missing (%)")
    ax.set_title("Missing Values by Column")
    ax.legend()
    fig.tight_layout()
    save(fig, "01_missing_values")


# ── 3. Disease Label Distribution ────────────────────────────────────────────

def plot_disease_distribution(df: pd.DataFrame) -> None:
    section("3. Disease Label Distribution")

    counts = {name: int(df[col].sum()) for name, col in DISEASE_COLS.items()}
    print(f"  Patient count: {len(df)}")
    for name, cnt in counts.items():
        print(f"  {name:<18}: {cnt:>3} ({100*cnt/len(df):.1f}%)")

    # Multi-label: how many diseases per patient
    label_matrix = df[[col for col in DISEASE_COLS.values()]].fillna(0)
    df["n_diseases"] = label_matrix.sum(axis=1)
    cooccurrence = df["n_diseases"].value_counts().sort_index()
    print(f"\n  Diseases per patient:\n{cooccurrence.to_string()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar – disease prevalence
    colors = sns.color_palette(PALETTE, len(counts))
    axes[0].bar(counts.keys(), counts.values(), color=colors, edgecolor="white")
    axes[0].set_ylabel("Patient Count")
    axes[0].set_title("Disease Prevalence (multi-label)")
    for i, (k, v) in enumerate(counts.items()):
        axes[0].text(i, v + 1, str(v), ha="center", fontsize=10, fontweight="bold")

    # Bar – co-occurrence
    axes[1].bar(cooccurrence.index.astype(str), cooccurrence.values,
                color=sns.color_palette("Blues_r", len(cooccurrence)), edgecolor="white")
    axes[1].set_xlabel("Number of Diseases per Patient")
    axes[1].set_ylabel("Patient Count")
    axes[1].set_title("Multi-Label Co-occurrence")
    for i, v in enumerate(cooccurrence.values):
        axes[1].text(i, v + 0.5, str(v), ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Disease Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "02_disease_distribution")


# ── 4. Demographics ───────────────────────────────────────────────────────────

def plot_demographics(df: pd.DataFrame) -> None:
    section("4. Demographics")

    age = pd.to_numeric(df["Âge (Age)"], errors="coerce").dropna()
    weight = pd.to_numeric(df["Poids (Weight)"], errors="coerce").dropna()
    gender = df["Genre (Gender)"].str.strip().value_counts()

    print(f"  Age   — mean: {age.mean():.1f}, median: {age.median():.1f}, "
          f"min: {age.min():.1f}, max: {age.max():.1f}")
    print(f"  Weight— mean: {weight.mean():.1f}, median: {weight.median():.1f}")
    print(f"  Gender: {gender.to_dict()}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Age histogram
    axes[0].hist(age, bins=20, color="#3498db", edgecolor="white")
    axes[0].axvline(age.median(), color="red", linestyle="--", label=f"Median={age.median():.0f}")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Age Distribution")
    axes[0].legend()

    # Age by disease (box plot)
    age_disease = []
    for name, col in DISEASE_COLS.items():
        mask = df[col] == 1
        ages_sub = pd.to_numeric(df.loc[mask, "Âge (Age)"], errors="coerce").dropna()
        age_disease.append(ages_sub.values)
    bp = axes[1].boxplot(age_disease, patch_artist=True, medianprops=dict(color="black", linewidth=2))
    colors = sns.color_palette(PALETTE, len(DISEASE_COLS))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    axes[1].set_xticklabels(DISEASE_COLS.keys(), rotation=25, ha="right")
    axes[1].set_ylabel("Age (years)")
    axes[1].set_title("Age Distribution per Disease")

    # Gender pie
    axes[2].pie(gender.values, labels=gender.index, autopct="%1.1f%%",
                colors=sns.color_palette(PALETTE, len(gender)), startangle=90,
                wedgeprops=dict(edgecolor="white"))
    axes[2].set_title("Gender Distribution")

    fig.suptitle("Patient Demographics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_demographics")


# ── 5. Symptom Prevalence ─────────────────────────────────────────────────────

def plot_symptom_prevalence(df: pd.DataFrame) -> None:
    section("5. Symptom Prevalence")

    sym_data = {}
    for label, col in SYMPTOM_COLS.items():
        if col in df.columns:
            binary = oui_non_to_binary(df[col].astype(str))
            sym_data[label] = binary.mean() * 100

    sym_series = pd.Series(sym_data).sort_values(ascending=True)
    print(f"  Top 5 symptoms:")
    for name, pct in sym_series.sort_values(ascending=False).head(5).items():
        print(f"    {name:<25}: {pct:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c" if v >= 50 else "#f39c12" if v >= 25 else "#3498db"
              for v in sym_series.values]
    ax.barh(sym_series.index, sym_series.values, color=colors)
    ax.set_xlabel("Prevalence (%)")
    ax.set_title("Symptom Prevalence Across All Patients")
    ax.axvline(50, color="red", linestyle="--", alpha=0.5, label="50%")
    ax.axvline(25, color="orange", linestyle="--", alpha=0.5, label="25%")
    ax.legend()
    for i, v in enumerate(sym_series.values):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=9)
    fig.tight_layout()
    save(fig, "04_symptom_prevalence")


# ── 6. Symptom vs Disease Heatmap ─────────────────────────────────────────────

def plot_symptom_disease_heatmap(df: pd.DataFrame) -> None:
    section("6. Symptom × Disease Heatmap")

    records = {}
    for sym_label, sym_col in SYMPTOM_COLS.items():
        if sym_col not in df.columns:
            continue
        sym_bin = oui_non_to_binary(df[sym_col].astype(str))
        row = {}
        for dis_label, dis_col in DISEASE_COLS.items():
            if dis_col not in df.columns:
                continue
            dis_bin = df[dis_col].fillna(0).astype(int)
            # Among patients with this disease, % having the symptom
            n_disease = dis_bin.sum()
            if n_disease > 0:
                row[dis_label] = (sym_bin[dis_bin == 1].sum() / n_disease) * 100
            else:
                row[dis_label] = 0
        records[sym_label] = row

    heatmap_df = pd.DataFrame(records).T  # shape: (symptoms × diseases)

    fig, ax = plt.subplots(figsize=(10, 11))
    sns.heatmap(
        heatmap_df,
        annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.4, linecolor="white",
        ax=ax, cbar_kws={"label": "% patients with symptom"}
    )
    ax.set_title("Symptom Prevalence (%) per Disease", fontsize=13, fontweight="bold")
    ax.set_xlabel("Disease")
    ax.set_ylabel("Symptom")
    fig.tight_layout()
    save(fig, "05_symptom_disease_heatmap")


# ── 7. Numeric / Lab Variables ────────────────────────────────────────────────

def plot_numeric_distributions(df: pd.DataFrame) -> None:
    section("7. Numeric & Lab Variables")

    available = {k: v for k, v in NUMERIC_COLS.items() if v in df.columns}
    print(f"  Numeric cols available: {list(available.keys())}")

    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, (label, col) in enumerate(available.items()):
        data = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(data) < 5:
            axes[i].set_visible(False)
            continue
        axes[i].hist(data, bins=25, color="#3498db", edgecolor="white", alpha=0.85)
        axes[i].axvline(data.median(), color="red", linestyle="--",
                        linewidth=1.5, label=f"Median={data.median():.1f}")
        axes[i].set_title(label, fontsize=10, fontweight="bold")
        axes[i].set_ylabel("Count")
        axes[i].legend(fontsize=8)
        axes[i].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric & Lab Variable Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "06_numeric_distributions")


# ── 8. Lab Values by Disease (Violin) ────────────────────────────────────────

def plot_lab_by_disease(df: pd.DataFrame) -> None:
    section("8. Lab Values by Disease")

    key_labs = {
        k: v for k, v in NUMERIC_COLS.items()
        if k in ["Temperature (°C)", "Hematocrit", "Platelets", "WBC"]
        and v in df.columns
    }

    # Build long-format: one disease label per patient (pick dominant)
    label_cols = [col for col in DISEASE_COLS.values() if col in df.columns]
    label_matrix = df[label_cols].fillna(0).astype(int)
    # Assign primary disease (first positive column; "Mixed" if >1; "None" if 0)
    def primary_disease(row):
        active = [name for name, col in DISEASE_COLS.items() if row.get(col, 0) == 1]
        if len(active) == 0:
            return "None"
        if len(active) == 1:
            return active[0]
        return "Mixed"

    df["primary_disease"] = df.apply(
        lambda r: primary_disease({col: r[col] for col in label_cols}), axis=1
    )

    order = ["Malaria", "Dengue", "Typhoid", "Yellow Fever", "Others", "Mixed"]
    order = [o for o in order if o in df["primary_disease"].unique()]

    n = len(key_labs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (label, col) in zip(axes, key_labs.items()):
        plot_df = df[["primary_disease", col]].copy()
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna(subset=[col])
        valid_order = [o for o in order if o in plot_df["primary_disease"].unique()]
        sns.violinplot(
            data=plot_df, x="primary_disease", y=col,
            order=valid_order, palette=PALETTE, ax=ax, inner="box"
        )
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Disease")
        ax.set_ylabel(label)
        ax.set_xticklabels(valid_order, rotation=30, ha="right")

    fig.suptitle("Key Lab Values by Primary Disease", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "07_lab_by_disease")


# ── 9. Comorbidities ──────────────────────────────────────────────────────────

def plot_comorbidities(df: pd.DataFrame) -> None:
    section("9. Comorbidities")

    com_data = {}
    for label, col in COMORBIDITY_COLS.items():
        if col in df.columns:
            binary = oui_non_to_binary(df[col].astype(str))
            pct = binary.mean() * 100
            com_data[label] = pct
            print(f"  {label:<20}: {pct:.1f}%")

    com_series = pd.Series(com_data).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(com_series.index, com_series.values,
            color=sns.color_palette("Oranges_r", len(com_series)))
    ax.set_xlabel("Prevalence (%)")
    ax.set_title("Comorbidity Prevalence")
    for i, v in enumerate(com_series.values):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=10)
    fig.tight_layout()
    save(fig, "08_comorbidities")


# ── 10. Numeric Correlation Matrix ───────────────────────────────────────────

def plot_correlation(df: pd.DataFrame) -> None:
    section("10. Correlation Matrix (Numeric)")

    num_df = pd.DataFrame()
    for label, col in NUMERIC_COLS.items():
        if col in df.columns:
            num_df[label] = pd.to_numeric(df[col], errors="coerce")

    corr = num_df.corr()
    print(f"  Correlation matrix shape: {corr.shape}")

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white", ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Correlation Matrix – Numeric & Lab Features", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "09_correlation_matrix")


# ── 11. Test Results Overview ─────────────────────────────────────────────────

def plot_test_results(df: pd.DataFrame) -> None:
    section("11. Diagnostic Test Results")

    test_cols = {
        "RDT (Test TDR)":       "Test TDR",
        "Thick Film (Goutte épaisse)": "Goutte épaisse ",
    }

    fig, axes = plt.subplots(1, len(test_cols), figsize=(10, 5))

    for ax, (label, col) in zip(axes, test_cols.items()):
        if col not in df.columns:
            continue
        counts = df[col].str.strip().value_counts().head(6)
        ax.bar(counts.index, counts.values,
               color=sns.color_palette(PALETTE, len(counts)), edgecolor="white")
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.5, str(v), ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Diagnostic Test Result Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "10_test_results")


# ── 12. Summary Report ────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    section("Summary")

    age = pd.to_numeric(df["Âge (Age)"], errors="coerce")
    gender = df["Genre (Gender)"].str.strip().value_counts()
    missing_pct = df.isnull().mean().mean() * 100

    print(f"""
  ┌─────────────────────────────────────────────────┐
  │           TRACK 4 – EDA SUMMARY                 │
  ├─────────────────────────────────────────────────┤
  │  Patients      : {len(df):<5}                         │
  │  Features      : {df.shape[1]:<5}                         │
  │  Missing cells : {missing_pct:.1f}%                        │
  │  Age (median)  : {age.median():.0f} years                     │
  │  Gender        : {gender.to_dict()}  │
  ├─────────────────────────────────────────────────┤
  │  Disease Prevalence                             │""")

    for name, col in DISEASE_COLS.items():
        if col in df.columns:
            cnt = int(df[col].sum())
            pct = 100 * cnt / len(df)
            print(f"  │  {name:<16}: {cnt:>3} patients ({pct:4.1f}%)              │")

    print("  └─────────────────────────────────────────────────┘")
    print(f"\n  All plots saved to: {OUTPUT_DIR.resolve()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    df = load_data(DATA_PATH)

    overview(df)
    plot_missing(df)
    plot_disease_distribution(df)
    plot_demographics(df)
    plot_symptom_prevalence(df)
    plot_symptom_disease_heatmap(df)
    plot_numeric_distributions(df)
    plot_lab_by_disease(df)
    plot_comorbidities(df)
    plot_correlation(df)
    plot_test_results(df)
    print_summary(df)


if __name__ == "__main__":
    main()