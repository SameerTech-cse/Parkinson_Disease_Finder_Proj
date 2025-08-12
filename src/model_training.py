from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import json
from joblib import dump
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
METRICS_PATH = Path(__file__).resolve().parents[1] / "reports" / "metrics.json"

def build_models(random_state=42):
    """Model zoo with search grids."""
    return {
        "log_reg": (
            LogisticRegression(max_iter=500),
            {"clf__C": [0.01, 0.1, 1, 10]}
        ),
        "svc": (
            SVC(probability=True),
            {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"]}
        ),
        "rf": (
            RandomForestClassifier(random_state=random_state),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 10, 20]}
        ),
        "dt": (
            DecisionTreeClassifier(random_state=random_state),
            {"clf__max_depth": [None, 10, 20]}
        ),
        "xgb": (
            XGBClassifier(
                random_state=random_state, eval_metric="logloss", n_jobs=-1
            ),
            {
                "clf__n_estimators": [200, 400],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 6],
            },
        ),
    }

def train_and_select_best(X_train, y_train, selector):
    """Train multiple models & return best estimator."""
    best_f1 = -np.inf
    best_model = None
    best_name = None
    best_metrics = {}

    for name, (clf, grid) in build_models().items():
        pipe = Pipeline(
            steps=[
                ("selector", selector),
                ("clf", clf),
            ]
        )
        search = GridSearchCV(
            pipe,
            grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="f1",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        if search.best_score_ > best_f1:
            best_f1 = search.best_score_
            best_model = search.best_estimator_
            best_name = name
        best_metrics[name] = {
            "f1_cv": search.best_score_,
            "best_params": search.best_params_,
        }

    # Persist results
    dump(best_model, MODEL_PATH)
    with open(METRICS_PATH, "w") as fp:
        json.dump(best_metrics, fp, indent=2)

    print(f"[INFO] Best model: {best_name} | f1_cv={best_f1:.3f}")
    return best_model, best_metrics
