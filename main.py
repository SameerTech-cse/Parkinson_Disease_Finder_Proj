from src.data_loader import load_data
from src.preprocessing import split_and_scale
from src.feature_selection import select_important_features
from src.model_training import train_and_select_best
from src.evaluation import evaluate
from pathlib import Path
import json

if __name__ == "__main__":
    print("[1] Loading data …")
    df = load_data()

    print(" Splitting, scaling & balancing …")
    X_train, X_test, y_train, y_test = split_and_scale(df)

    print(" Feature selection …")
    selector = select_important_features(X_train, y_train)

    print(" Model training & hyper-parameter tuning …")
    best_model, cv_metrics = train_and_select_best(X_train, y_train, selector)

    print(" Final evaluation …")
    test_metrics = evaluate(best_model, X_test, y_test)
    print("Test metrics →", test_metrics)

    # Append test metrics to report
    rpt_path = Path("reports/metrics.json")
    if rpt_path.exists():
        with open(rpt_path) as fp:
            all_metrics = json.load(fp)
    else:
        all_metrics = {}
    all_metrics["best_model_test"] = test_metrics
    with open(rpt_path, "w") as fp:
        json.dump(all_metrics, fp, indent=2)

    print("[✓] Pipeline complete. Launching Streamlit app …")
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app.py"])
