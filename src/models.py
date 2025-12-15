"""
Student Depression Dataset - Modeling Module
=============================================
Functions for model training, evaluation, and comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
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
)


# =============================================================================
# DATA SPLITTING
# =============================================================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
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
    return X_train, X_test, y_train, y_test


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_default_models() -> Dict[str, Any]:
    """
    Get dictionary of default models to train.

    Returns
    -------
    Dict[str, Any]
        Dictionary of model name -> model instance.
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5)
    }
    return models


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train a single model.

    Parameters
    ----------
    model : sklearn estimator
        Model to train.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.

    Returns
    -------
    trained model
    """
    model.fit(X_train, y_train)
    return model


def train_all_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_train_scaled: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Train all models.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of models to train.
    X_train : pd.DataFrame
        Training features (unscaled).
    y_train : pd.Series
        Training target.
    X_train_scaled : pd.DataFrame, optional
        Scaled training features for Logistic Regression.

    Returns
    -------
    Dict[str, Any]
        Dictionary of trained models.
    """
    trained_models = {}

    for name, model in models.items():
        # Use scaled features for Logistic Regression
        if name == 'Logistic Regression' and X_train_scaled is not None:
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.

    Parameters
    ----------
    model : trained model
        Model with predict and predict_proba methods.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels.
    model_name : str
        Name of the model.

    Returns
    -------
    Dict[str, float]
        Dictionary of metric name -> value.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob),
        'F1': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }

    return metrics


def evaluate_all_models(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_test_scaled: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Evaluate all trained models.

    Parameters
    ----------
    trained_models : Dict[str, Any]
        Dictionary of trained models.
    X_test : pd.DataFrame
        Test features (unscaled).
    y_test : pd.Series
        Test target.
    X_test_scaled : pd.DataFrame, optional
        Scaled test features for Logistic Regression.

    Returns
    -------
    pd.DataFrame
        DataFrame with evaluation metrics for all models.
    """
    results = []

    for name, model in trained_models.items():
        # Use scaled features for Logistic Regression
        if name == 'Logistic Regression' and X_test_scaled is not None:
            metrics = evaluate_model(model, X_test_scaled, y_test, name)
        else:
            metrics = evaluate_model(model, X_test, y_test, name)

        metrics['Model'] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    cols = ['Model', 'Accuracy', 'AUC', 'F1', 'Precision', 'Recall']
    results_df = results_df[cols]

    return results_df


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = 'roc_auc'
) -> Dict[str, float]:
    """
    Perform cross-validation for a single model.

    Parameters
    ----------
    model : sklearn estimator
        Model to cross-validate.
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target.
    cv : int
        Number of folds.
    scoring : str
        Scoring metric.

    Returns
    -------
    Dict[str, float]
        Dictionary with mean and std of CV scores.
    """
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring)

    return {
        'mean': scores.mean(),
        'std': scores.std()
    }


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(
    model,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Get feature importance from a trained model.

    Parameters
    ----------
    model : trained model
        Model with feature_importances_ or coef_ attribute.
    feature_names : List[str]
        List of feature names.

    Returns
    -------
    pd.DataFrame
        DataFrame with features and their importance scores.
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    return importance_df


# =============================================================================
# ROC CURVE DATA
# =============================================================================

def get_roc_curve_data(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Get ROC curve data for plotting.

    Parameters
    ----------
    model : trained model
        Model with predict_proba method.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        (fpr, tpr, auc_score)
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    return fpr, tpr, auc


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def get_confusion_matrix(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> np.ndarray:
    """
    Get confusion matrix for a model.

    Parameters
    ----------
    model : trained model
        Model with predict method.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.

    Returns
    -------
    np.ndarray
        Confusion matrix.
    """
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


# =============================================================================
# CLASSIFICATION REPORT
# =============================================================================

def get_classification_report(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: List[str] = None
) -> str:
    """
    Get classification report for a model.

    Parameters
    ----------
    model : trained model
        Model with predict method.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    target_names : List[str], optional
        Names for target classes.

    Returns
    -------
    str
        Classification report string.
    """
    if target_names is None:
        target_names = ['No Depression', 'Depression']

    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, target_names=target_names)


# =============================================================================
# LOGISTIC REGRESSION COEFFICIENTS
# =============================================================================

def get_logistic_coefficients(
    model: LogisticRegression,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Get coefficients and odds ratios from logistic regression.

    Parameters
    ----------
    model : LogisticRegression
        Trained logistic regression model.
    feature_names : List[str]
        List of feature names.

    Returns
    -------
    pd.DataFrame
        DataFrame with coefficients and odds ratios.
    """
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0],
        'Odds Ratio': np.exp(model.coef_[0])
    })

    coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)

    return coef_df


# =============================================================================
# SCALE FEATURES
# =============================================================================

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]
        (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler
