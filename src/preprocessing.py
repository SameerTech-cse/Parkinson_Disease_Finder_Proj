import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from joblib import dump
from pathlib import Path

SCALER_PATH = Path(__file__).resolve().parents[1] / "models" / "scaler.joblib"

def split_and_scale(
    df: pd.DataFrame,
    target: str = "class",
    test_size: float = 0.2,
    random_state: int = 42
):
    """Handle class imbalance, split data, scale features."""
    X = df.drop(columns=[target])
    y = df[target]

    # SMOTE to correct imbalance
    X_res, y_res = SMOTE(random_state=random_state).fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=test_size, stratify=y_res, random_state=random_state
    )

    # Standardisation
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    scaler = StandardScaler().fit(X_train_selected)
    X_train_scaled = scaler.transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test)
    dump(scaler, SCALER_PATH)

    return X_train_scaled, X_test_scaled, y_train, y_test
