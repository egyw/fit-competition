import pandas as pd
import numpy as np
from pathlib import Path


# load data from excel
def load_data(sheet_name=None, skiprows=1):
    """Load the Track 5 workbook from the raw data directory.

    Args:
        sheet_name: Sheet index/name passed to pandas.read_excel.
            Use None to load all five sheets.
        skiprows: Number of leading rows to skip in each sheet.

    Returns:
        pandas.DataFrame or dict[str, pandas.DataFrame]
    """
    project_root = Path(__file__).resolve().parents[2]

    # First try the path requested by the user.
    requested_path = (
        project_root
        / "data"
        / "raw"
        / "Track 5"
        / "Rice Supply Chain in West Java Province, Indonesia.xlsx"
    )

    if requested_path.exists():
        return pd.read_excel(requested_path, sheet_name=sheet_name, skiprows=skiprows)

    # Fallback for the current folder structure where the file is nested.
    track_5_dir = project_root / "data" / "raw" / "Track 5"
    matches = list(
        track_5_dir.rglob("Rice Supply Chain in West Java Province, Indonesia.xlsx")
    )
    if not matches:
        raise FileNotFoundError(
            "Track 5 dataset not found in data/raw/Track 5. "
            "Expected 'Rice Supply Chain in West Java Province, Indonesia.xlsx'."
        )

    return pd.read_excel(matches[0], sheet_name=sheet_name, skiprows=skiprows)

if __name__ == "__main__":
    sheets = load_data()
    for name, df in sheets.items():
        print(f"{name}: {df.shape}")