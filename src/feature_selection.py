from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def select_important_features(
    X, y, threshold: str = "median", random_state: int = 42
):
    """Select features using embedded RF importance."""
    selector = SelectFromModel(
        RandomForestClassifier(
            n_estimators=400, random_state=random_state, n_jobs=-1
        ),
        threshold=threshold,
    )
    selector.fit(X, y)
    return selector
