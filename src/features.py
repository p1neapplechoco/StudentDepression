"""
Student Depression Dataset - Feature Engineering Module
=======================================================
Functions for creating features for modeling.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler


# =============================================================================
# AGE GROUPING
# =============================================================================

def create_age_groups(df: pd.DataFrame,
                      bins: List[int] = None,
                      labels: List[str] = None) -> pd.DataFrame:
    """
    Create age group categories.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Age column.
    bins : List[int], optional
        Age bin boundaries. Default: [17, 20, 23, 26, 30, 35, 40]
    labels : List[str], optional
        Labels for age groups. Default: ['18-20', '21-23', '24-26', '27-30', '31-35', '36+']

    Returns
    -------
    pd.DataFrame
        Dataset with Age_Group column added.
    """
    df_featured = df.copy()

    if bins is None:
        bins = [17, 20, 23, 26, 30, 35, 40]
    if labels is None:
        labels = ['18-20', '21-23', '24-26', '27-30', '31-35', '36+']

    if 'Age' in df_featured.columns:
        df_featured['Age_Group'] = pd.cut(df_featured['Age'], bins=bins, labels=labels)

    return df_featured


# =============================================================================
# CGPA GROUPING
# =============================================================================

def create_cgpa_groups(df: pd.DataFrame,
                       bins: List[float] = None,
                       labels: List[str] = None) -> pd.DataFrame:
    """
    Create CGPA group categories.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with CGPA column.
    bins : List[float], optional
        CGPA bin boundaries. Default: [0, 5, 6, 7, 8, 9, 11]
    labels : List[str], optional
        Labels for CGPA groups. Default: ['<5', '5-6', '6-7', '7-8', '8-9', '9+']

    Returns
    -------
    pd.DataFrame
        Dataset with CGPA_Group column added.
    """
    df_featured = df.copy()

    if bins is None:
        bins = [0, 5, 6, 7, 8, 9, 11]
    if labels is None:
        labels = ['<5', '5-6', '6-7', '7-8', '8-9', '9+']

    if 'CGPA' in df_featured.columns:
        df_featured['CGPA_Group'] = pd.cut(df_featured['CGPA'], bins=bins, labels=labels)

    return df_featured


# =============================================================================
# STUDY SATISFACTION GROUPING
# =============================================================================

def create_satisfaction_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Study Satisfaction group categories.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Study Satisfaction column.

    Returns
    -------
    pd.DataFrame
        Dataset with Satisfaction_Group column added.
    """
    df_featured = df.copy()

    if 'Study Satisfaction' in df_featured.columns:
        df_featured['Satisfaction_Group'] = pd.cut(
            df_featured['Study Satisfaction'],
            bins=[0, 2, 3, 5],
            labels=['Low (1-2)', 'Medium (3)', 'High (4-5)']
        )

    return df_featured


# =============================================================================
# ACADEMIC PRESSURE GROUPING
# =============================================================================

def create_pressure_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Academic Pressure group categories.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Academic Pressure column.

    Returns
    -------
    pd.DataFrame
        Dataset with AP_Group column added.
    """
    df_featured = df.copy()

    if 'Academic Pressure' in df_featured.columns:
        df_featured['AP_Group'] = pd.cut(
            df_featured['Academic Pressure'],
            bins=[0, 2, 4, 6],
            labels=['Low (1-2)', 'Medium (3-4)', 'High (5)']
        )

    return df_featured


# =============================================================================
# WORK/STUDY HOURS GROUPING
# =============================================================================

def create_hours_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Work/Study Hours group categories.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Work/Study Hours column.

    Returns
    -------
    pd.DataFrame
        Dataset with Hours_Group column added.
    """
    df_featured = df.copy()

    if 'Work/Study Hours' in df_featured.columns:
        df_featured['Hours_Group'] = pd.cut(
            df_featured['Work/Study Hours'],
            bins=[0, 3, 6, 9, 15],
            labels=['0-3h', '3-6h', '6-9h', '9h+']
        )

    return df_featured


# =============================================================================
# INTERACTION FEATURES
# =============================================================================

def create_interaction_ap_ss(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction term between Academic Pressure and Study Satisfaction.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Academic Pressure and Study Satisfaction columns.

    Returns
    -------
    pd.DataFrame
        Dataset with AP_SS_Interaction column added.
    """
    df_featured = df.copy()

    if 'Academic Pressure' in df_featured.columns and 'Study Satisfaction' in df_featured.columns:
        df_featured['AP_SS_Interaction'] = (
            df_featured['Academic Pressure'] * df_featured['Study Satisfaction']
        )

    return df_featured


# =============================================================================
# ACHIEVEMENT-SATISFACTION COMBINATION
# =============================================================================

def create_achievement_satisfaction_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined Achievement (CGPA) and Satisfaction category.

    Categories:
    - High CGPA + Low Satisfaction
    - High CGPA + High Satisfaction
    - Low CGPA + Low Satisfaction
    - Low CGPA + High Satisfaction
    - Medium

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with CGPA and Study Satisfaction columns.

    Returns
    -------
    pd.DataFrame
        Dataset with Achievement_Sat column added.
    """
    df_featured = df.copy()

    if 'CGPA' in df_featured.columns and 'Study Satisfaction' in df_featured.columns:
        def categorize(row):
            cgpa = row['CGPA']
            sat = row['Study Satisfaction']

            if cgpa >= 8 and sat <= 2:
                return 'High CGPA + Low Satisfaction'
            elif cgpa >= 8 and sat >= 4:
                return 'High CGPA + High Satisfaction'
            elif cgpa < 8 and sat <= 2:
                return 'Low CGPA + Low Satisfaction'
            elif cgpa < 8 and sat >= 4:
                return 'Low CGPA + High Satisfaction'
            else:
                return 'Medium'

        df_featured['Achievement_Sat'] = df_featured.apply(categorize, axis=1)

    return df_featured


# =============================================================================
# HIGH ACHIEVER FLAG
# =============================================================================

def create_high_achiever_flag(df: pd.DataFrame, threshold: float = 8.0) -> pd.DataFrame:
    """
    Create flag for high achievers based on CGPA.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with CGPA column.
    threshold : float
        CGPA threshold for high achiever classification.

    Returns
    -------
    pd.DataFrame
        Dataset with High_Achiever column added.
    """
    df_featured = df.copy()

    if 'CGPA' in df_featured.columns:
        df_featured['High_Achiever'] = (df_featured['CGPA'] >= threshold).astype(int)

    return df_featured


# =============================================================================
# MODEL FEATURE PREPARATION
# =============================================================================

def prepare_model_features(df: pd.DataFrame,
                           features: List[str] = None,
                           target: str = 'Depression',
                           scale: bool = False) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Prepare features for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with all features.
    features : List[str], optional
        List of feature column names. If None, uses default features.
    target : str
        Target column name.
    scale : bool
        Whether to scale features.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, StandardScaler]
        (X, y, scaler) - Features, target, and fitted scaler (None if scale=False).
    """
    if features is None:
        features = [
            'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
            'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours',
            'Financial Stress', 'Sleep_Hours_Encoded', 'Diet_Encoded',
            'Suicidal_Thoughts', 'Family_History', 'Gender_Encoded'
        ]

    # Filter to available features
    available_features = [f for f in features if f in df.columns]

    # Prepare data (drop rows with missing values in selected columns)
    model_df = df[available_features + [target]].dropna()

    X = model_df[available_features]
    y = model_df[target]

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X, y, scaler


def get_default_model_features() -> List[str]:
    """
    Get default list of features for modeling.

    Returns
    -------
    List[str]
        List of default feature column names.
    """
    return [
        'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
        'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours',
        'Financial Stress', 'Sleep_Hours_Encoded', 'Diet_Encoded',
        'Suicidal_Thoughts', 'Family_History', 'Gender_Encoded'
    ]


# =============================================================================
# APPLY ALL FEATURE ENGINEERING
# =============================================================================

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to engineer.

    Returns
    -------
    pd.DataFrame
        Dataset with all engineered features added.
    """
    df_featured = df.copy()

    df_featured = create_age_groups(df_featured)
    df_featured = create_cgpa_groups(df_featured)
    df_featured = create_satisfaction_groups(df_featured)
    df_featured = create_pressure_groups(df_featured)
    df_featured = create_hours_groups(df_featured)
    df_featured = create_interaction_ap_ss(df_featured)
    df_featured = create_achievement_satisfaction_category(df_featured)
    df_featured = create_high_achiever_flag(df_featured)

    return df_featured
