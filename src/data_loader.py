import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "parkinson_disease_Updated.csv"

def load_data() -> pd.DataFrame:
    """Load Parkinson dataset."""
    df = pd.read_csv(DATA_PATH)
    # Basic sanity checks
    if df.isnull().sum().any():
        print(f"[WARN] Missing values detected. Rows: {df.shape}")
    return df