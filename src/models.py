import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

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


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_default_models() -> Dict[str, Any]:
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5)
    }
    return models


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    model.fit(X_train, y_train)
    return model


def train_all_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_train_scaled: pd.DataFrame = None
) -> Dict[str, Any]:
    trained_models = {}

    for name, model in models.items():
        if name == 'Logistic Regression' and X_train_scaled is not None:
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, float]:
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
    results = []

    for name, model in trained_models.items():
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


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = 'roc_auc'
) -> Dict[str, float]:
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring)

    return {
        'mean': scores.mean(),
        'std': scores.std()
    }


def get_feature_importance(
    model,
    feature_names: List[str]
) -> pd.DataFrame:
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


def get_roc_curve_data(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[np.ndarray, np.ndarray, float]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    return fpr, tpr, auc


def get_confusion_matrix(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> np.ndarray:
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def get_classification_report(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: List[str] = None
) -> str:
    if target_names is None:
        target_names = ['No Depression', 'Depression']

    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, target_names=target_names)


def get_logistic_coefficients(
    model: LogisticRegression,
    feature_names: List[str]
) -> pd.DataFrame:
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0],
        'Odds Ratio': np.exp(model.coef_[0])
    })

    coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)

    return coef_df


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
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


def modeling_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    use_smote: bool = False,
    cross_validate: bool = True
) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    models = get_default_models()
    trained_models = train_all_models(models, X_train, y_train, X_train_scaled)
    results_df = evaluate_all_models(trained_models, X_test, y_test, X_test_scaled)

    best_model_name = results_df.loc[results_df['AUC'].idxmax(), 'Model']
    best_model = trained_models[best_model_name]

    feature_importance = None
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        feature_importance = get_feature_importance(best_model, X.columns.tolist())

    results_df = results_df.set_index('Model')

    return {
        'results': results_df,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'trained_models': trained_models,
        'feature_importance': feature_importance,
        'scaler': scaler
    }


def save_results(results_df: pd.DataFrame, filepath: str):
    results_df.to_csv(filepath)
