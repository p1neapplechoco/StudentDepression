"""
Student Depression Dataset - Modeling Module
=============================================
Functions for model training, evaluation, and comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")

# Scikit-learn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

# Try to import optional libraries
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


# =============================================================================
# DATA SPLITTING
# =============================================================================


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with stratification.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    test_size : float
        Proportion of data for test set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"✅ Data split:")
    print(f"   Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"   Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"   Test class distribution: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE oversampling to handle class imbalance.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    random_state : int
        Random seed.

    Returns
    -------
    Tuple
        (X_resampled, y_resampled)
    """
    if not IMBLEARN_AVAILABLE:
        print("⚠️ imbalanced-learn not installed. Skipping SMOTE.")
        return X_train, y_train

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"✅ SMOTE applied:")
    print(f"   Before: {len(X_train)} samples")
    print(f"   After: {len(X_resampled)} samples")
    print(
        f"   New class distribution: {pd.Series(y_resampled).value_counts().to_dict()}"
    )

    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================


def get_models(use_class_weight: bool = True) -> Dict[str, Any]:
    """
    Get dictionary of models to train.

    Parameters
    ----------
    use_class_weight : bool
        Whether to use class weighting for imbalance.

    Returns
    -------
    Dict[str, Any]
        Dictionary of model name -> model instance.
    """
    class_weight = "balanced" if use_class_weight else None

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight=class_weight, max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight=class_weight, max_depth=10, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            class_weight=class_weight,
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
    }

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )

    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
        )

    return models


# =============================================================================
# MODEL EVALUATION
# =============================================================================


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.

    Parameters
    ----------
    model : Any
        Trained model with predict and predict_proba methods.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels.
    model_name : str
        Name of the model for display.

    Returns
    -------
    Dict[str, float]
        Dictionary of metric name -> value.
    """
    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities (if available)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        has_proba = True
    except:
        y_prob = None
        has_proba = False

    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
    }

    if has_proba:
        metrics["AUC-ROC"] = roc_auc_score(y_test, y_prob)

    return metrics


def print_evaluation_report(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model"
) -> None:
    """
    Print detailed evaluation report for a model.

    Parameters
    ----------
    model : Any
        Trained model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels.
    model_name : str
        Name of the model.
    """
    y_pred = model.predict(X_test)

    print(f"\n{'='*60}")
    print(f"📊 {model_name} - Evaluation Report")
    print("=" * 60)

    print("\n📈 Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["No Depression", "Depression"]
        )
    )

    print("\n📉 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   TN: {cm[0,0]:<6} FP: {cm[0,1]:<6}")
    print(f"   FN: {cm[1,0]:<6} TP: {cm[1,1]:<6}")


# =============================================================================
# MODEL TRAINING & COMPARISON
# =============================================================================


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models: Dict[str, Any] = None,
    scale_features: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train and evaluate multiple models.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training target.
    y_test : pd.Series
        Test target.
    models : Dict[str, Any]
        Dictionary of models to train. If None, uses default models.
    scale_features : bool
        Whether to scale features (recommended for Logistic Regression).

    Returns
    -------
    Tuple[Dict[str, Any], pd.DataFrame]
        (trained_models, results_df)
    """
    if models is None:
        models = get_models()

    # Scale features if requested
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        scaler = None

    trained_models = {}
    results = []

    print("=" * 60)
    print("🚀 TRAINING MODELS")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n🔄 Training {name}...")

        try:
            # Train
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model

            # Evaluate
            metrics = evaluate_model(model, X_test_scaled, y_test, name)
            metrics["Model"] = name
            results.append(metrics)

            print(f"   ✅ Accuracy: {metrics['Accuracy']:.4f}")
            print(f"   ✅ Recall: {metrics['Recall']:.4f}")
            print(f"   ✅ F1-Score: {metrics['F1-Score']:.4f}")
            if "AUC-ROC" in metrics:
                print(f"   ✅ AUC-ROC: {metrics['AUC-ROC']:.4f}")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index("Model")
    results_df = results_df.sort_values("F1-Score", ascending=False)

    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON (sorted by F1-Score)")
    print("=" * 60)
    print(results_df.round(4).to_string())

    return trained_models, results_df, scaler


# =============================================================================
# CROSS-VALIDATION
# =============================================================================


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Any] = None,
    cv: int = 5,
    scoring: str = "f1",
) -> pd.DataFrame:
    """
    Perform cross-validation for multiple models.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target.
    models : Dict[str, Any]
        Dictionary of models. If None, uses default models.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric for cross-validation.

    Returns
    -------
    pd.DataFrame
        Cross-validation results.
    """
    if models is None:
        models = get_models()

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    results = []

    print("=" * 60)
    print(f"🔄 CROSS-VALIDATION ({cv}-Fold, scoring={scoring})")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n⏳ {name}...")

        try:
            scores = cross_val_score(
                model, X_scaled, y, cv=cv_strategy, scoring=scoring, n_jobs=-1
            )

            results.append(
                {
                    "Model": name,
                    "Mean Score": scores.mean(),
                    "Std Score": scores.std(),
                    "Min Score": scores.min(),
                    "Max Score": scores.max(),
                }
            )

            print(f"   ✅ {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

    results_df = pd.DataFrame(results)
    results_df = results_df.set_index("Model")
    results_df = results_df.sort_values("Mean Score", ascending=False)

    print("\n" + "=" * 60)
    print("📊 CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(results_df.round(4).to_string())

    return results_df


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================


def get_feature_importance(
    model: Any, feature_names: List[str], model_name: str = "Model"
) -> pd.DataFrame:
    """
    Get feature importance from a trained model.

    Parameters
    ----------
    model : Any
        Trained model.
    feature_names : List[str]
        List of feature names.
    model_name : str
        Name of the model.

    Returns
    -------
    pd.DataFrame
        Feature importance DataFrame.
    """
    # Try to get feature importance
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        print(f"⚠️ {model_name} does not support feature importance")
        return None

    # Create DataFrame
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    importance_df = importance_df.sort_values("Importance", ascending=False)
    importance_df["Rank"] = range(1, len(importance_df) + 1)

    return importance_df


def print_top_features(
    importance_df: pd.DataFrame, top_n: int = 10, model_name: str = "Model"
) -> None:
    """
    Print top N most important features.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance DataFrame.
    top_n : int
        Number of top features to display.
    model_name : str
        Name of the model.
    """
    print(f"\n{'='*60}")
    print(f"🏆 Top {top_n} Features - {model_name}")
    print("=" * 60)

    for i, row in importance_df.head(top_n).iterrows():
        bar_len = int(row["Importance"] / importance_df["Importance"].max() * 30)
        bar = "█" * bar_len
        print(
            f"   {row['Rank']:>2}. {row['Feature']:<35} | {bar} {row['Importance']:.4f}"
        )


# =============================================================================
# SAVE/LOAD UTILITIES
# =============================================================================


def save_results(results_df: pd.DataFrame, filepath: str) -> None:
    """
    Save model comparison results to CSV.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame.
    filepath : str
        Output file path.
    """
    results_df.to_csv(filepath)
    print(f"✅ Results saved to {filepath}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def modeling_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    use_smote: bool = False,
    cross_validate: bool = True,
) -> Dict[str, Any]:
    """
    Run the full modeling pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target.
    test_size : float
        Proportion of data for test set.
    use_smote : bool
        Whether to use SMOTE for class imbalance.
    cross_validate : bool
        Whether to perform cross-validation.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing trained models, results, and best model.
    """
    print("=" * 60)
    print("🚀 MODELING PIPELINE")
    print("=" * 60)

    # Step 1: Split data
    print("\n📊 Step 1: Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)

    # Step 2: Handle class imbalance (optional)
    if use_smote:
        print("\n⚖️ Step 2: Applying SMOTE...")
        X_train, y_train = apply_smote(X_train, y_train)

    # Step 3: Cross-validation (optional)
    if cross_validate:
        print("\n🔄 Step 3: Cross-validation...")
        cv_results = cross_validate_models(X_train, y_train)
    else:
        cv_results = None

    # Step 4: Train and evaluate models
    print("\n🎯 Step 4: Training and evaluating models...")
    trained_models, results_df, scaler = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )

    # Step 5: Get best model
    best_model_name = results_df.index[0]
    best_model = trained_models[best_model_name]

    print("\n" + "=" * 60)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print("=" * 60)

    # Print detailed report for best model
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    print_evaluation_report(best_model, X_test_scaled, y_test, best_model_name)

    # Step 6: Feature importance for best model
    print("\n📈 Step 6: Feature importance analysis...")
    importance_df = get_feature_importance(
        best_model, X_train.columns.tolist(), best_model_name
    )
    if importance_df is not None:
        print_top_features(importance_df, top_n=10, model_name=best_model_name)

    return {
        "trained_models": trained_models,
        "results": results_df,
        "cv_results": cv_results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "scaler": scaler,
        "feature_importance": importance_df,
        "splits": (X_train, X_test, y_train, y_test),
    }


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    import os
    import sys

    # Add src to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    from preprocessing import preprocess_pipeline
    from features import feature_engineering_pipeline, prepare_modeling_data

    # Get project root
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "student_depression_dataset.csv")

    # Run preprocessing
    df_processed, _ = preprocess_pipeline(data_path)

    # Run feature engineering
    df_features = feature_engineering_pipeline(df_processed)

    # Prepare data for modeling
    X, y, feature_cols = prepare_modeling_data(df_features)

    # Run modeling pipeline
    results = modeling_pipeline(X, y, use_smote=False, cross_validate=True)

    print("\n✅ Modeling pipeline complete!")
