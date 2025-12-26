import os
import sys
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from preprocessing import preprocess_pipeline
from features import feature_engineering_pipeline, prepare_modeling_data
from models import modeling_pipeline, save_results

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "student_depression_dataset.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

USE_SMOTE = False
CROSS_VALIDATE = True
TEST_SIZE = 0.2


def main():
    df_processed, _ = preprocess_pipeline(DATA_PATH)
    df_features = feature_engineering_pipeline(df_processed)
    X, y, _ = prepare_modeling_data(df_features)
    results = modeling_pipeline(
        X, y, test_size=TEST_SIZE, use_smote=USE_SMOTE, cross_validate=CROSS_VALIDATE
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    save_results(results["results"], results_path)

    if results["feature_importance"] is not None:
        importance_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
        results["feature_importance"].to_csv(importance_path, index=False)

    processed_path = os.path.join(RESULTS_DIR, "processed_data.csv")
    df_features.to_csv(processed_path, index=False)

    return results


if __name__ == "__main__":
    results = main()
