#!/usr/bin/env python3
"""
Student Depression Analysis - Main Pipeline
============================================
Complete pipeline from raw data to model evaluation.

Usage:
    python run_pipeline.py

Output:
    - Model comparison results
    - Feature importance analysis
    - Best model evaluation report
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# Import modules
from preprocessing import preprocess_pipeline
from features import feature_engineering_pipeline, prepare_modeling_data
from models import modeling_pipeline, save_results

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "student_depression_dataset.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Pipeline settings
USE_SMOTE = False  # Set to True to handle class imbalance with SMOTE
CROSS_VALIDATE = True  # Set to True to perform cross-validation
TEST_SIZE = 0.2  # Proportion of data for test set


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def main():
    """Run the complete analysis pipeline."""

    print("\n" + "=" * 70)
    print("🎓 STUDENT DEPRESSION PREDICTION - COMPLETE PIPELINE")
    print("=" * 70)
    print(f"\n📁 Data path: {DATA_PATH}")
    print(f"⚙️  Settings: SMOTE={USE_SMOTE}, CV={CROSS_VALIDATE}, test_size={TEST_SIZE}")

    # =========================================================================
    # PHASE 1: PREPROCESSING
    # =========================================================================
    print("\n\n" + "▓" * 70)
    print("▓ PHASE 1: DATA PREPROCESSING")
    print("▓" * 70)

    df_processed, df_original = preprocess_pipeline(DATA_PATH)

    print(f"\n📊 Original shape: {df_original.shape}")
    print(f"📊 Processed shape: {df_processed.shape}")

    # =========================================================================
    # PHASE 2: FEATURE ENGINEERING
    # =========================================================================
    print("\n\n" + "▓" * 70)
    print("▓ PHASE 2: FEATURE ENGINEERING")
    print("▓" * 70)

    df_features = feature_engineering_pipeline(df_processed)

    print(
        f"\n📊 Features added: {df_features.shape[1] - df_processed.shape[1]} new columns"
    )

    # =========================================================================
    # PHASE 3: PREPARE MODELING DATA
    # =========================================================================
    print("\n\n" + "▓" * 70)
    print("▓ PHASE 3: PREPARING MODELING DATA")
    print("▓" * 70)

    X, y, feature_cols = prepare_modeling_data(df_features)

    print(f"\n📊 Features used for modeling: {len(feature_cols)}")
    print(f"📊 Target distribution:")
    print(f"   - No Depression (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"   - Depression (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

    # =========================================================================
    # PHASE 4: MODELING
    # =========================================================================
    print("\n\n" + "▓" * 70)
    print("▓ PHASE 4: MODEL TRAINING & EVALUATION")
    print("▓" * 70)

    results = modeling_pipeline(
        X, y, test_size=TEST_SIZE, use_smote=USE_SMOTE, cross_validate=CROSS_VALIDATE
    )

    # =========================================================================
    # PHASE 5: SAVE RESULTS
    # =========================================================================
    print("\n\n" + "▓" * 70)
    print("▓ PHASE 5: SAVING RESULTS")
    print("▓" * 70)

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save model comparison results
    results_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    save_results(results["results"], results_path)

    # Save feature importance
    if results["feature_importance"] is not None:
        importance_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
        results["feature_importance"].to_csv(importance_path, index=False)
        print(f"✅ Feature importance saved to {importance_path}")

    # Save processed data
    processed_path = os.path.join(RESULTS_DIR, "processed_data.csv")
    df_features.to_csv(processed_path, index=False)
    print(f"✅ Processed data saved to {processed_path}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 70)

    print(f"\n📊 Best Model: {results['best_model_name']}")
    best_metrics = results["results"].loc[results["best_model_name"]]
    print(f"   • Accuracy: {best_metrics['Accuracy']:.4f}")
    print(f"   • Precision: {best_metrics['Precision']:.4f}")
    print(f"   • Recall: {best_metrics['Recall']:.4f}")
    print(f"   • F1-Score: {best_metrics['F1-Score']:.4f}")
    if "AUC-ROC" in best_metrics:
        print(f"   • AUC-ROC: {best_metrics['AUC-ROC']:.4f}")

    print(f"\n📁 Results saved to: {RESULTS_DIR}/")
    print("   • model_comparison.csv")
    print("   • feature_importance.csv")
    print("   • processed_data.csv")

    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = main()
