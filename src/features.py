"""
Student Depression Dataset - Feature Engineering Module
=======================================================
Functions for creating new features from preprocessed data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


# =============================================================================
# COMPOSITE FEATURES
# =============================================================================


def create_lifestyle_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Lifestyle Score by combining Sleep and Diet encoded values.
    Higher score = healthier lifestyle.

    Score Range: 0-5 (Sleep: 0-3, Diet: 0-2)

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset with Sleep_Encoded and Diet_Encoded columns.

    Returns
    -------
    pd.DataFrame
        Dataset with Lifestyle_Score column added.
    """
    df_feat = df.copy()

    if "Sleep_Encoded" in df_feat.columns and "Diet_Encoded" in df_feat.columns:
        df_feat["Lifestyle_Score"] = df_feat["Sleep_Encoded"] + df_feat["Diet_Encoded"]
        print(
            f"✅ Created Lifestyle_Score (range: {df_feat['Lifestyle_Score'].min()}-{df_feat['Lifestyle_Score'].max()})"
        )
    else:
        print("⚠️ Missing Sleep_Encoded or Diet_Encoded columns")

    return df_feat


def create_total_stress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Total Stress Score by combining Academic and Financial stress.

    Score Range: 0-10 (each stress is 0-5)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Academic Pressure and Financial Stress columns.

    Returns
    -------
    pd.DataFrame
        Dataset with Total_Stress column added.
    """
    df_feat = df.copy()

    if "Academic Pressure" in df_feat.columns and "Financial Stress" in df_feat.columns:
        df_feat["Total_Stress"] = (
            df_feat["Academic Pressure"] + df_feat["Financial Stress"]
        )
        print(
            f"✅ Created Total_Stress (range: {df_feat['Total_Stress'].min()}-{df_feat['Total_Stress'].max()})"
        )
    else:
        print("⚠️ Missing Academic Pressure or Financial Stress columns")

    return df_feat


def create_study_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Study Efficiency = CGPA / (Work/Study Hours + 1).
    Higher = more efficient (high grades with less hours).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with CGPA and Work/Study Hours columns.

    Returns
    -------
    pd.DataFrame
        Dataset with Study_Efficiency column added.
    """
    df_feat = df.copy()

    if "CGPA" in df_feat.columns and "Work/Study Hours" in df_feat.columns:
        df_feat["Study_Efficiency"] = df_feat["CGPA"] / (
            df_feat["Work/Study Hours"] + 1
        )
        print(
            f"✅ Created Study_Efficiency (range: {df_feat['Study_Efficiency'].min():.2f}-{df_feat['Study_Efficiency'].max():.2f})"
        )
    else:
        print("⚠️ Missing CGPA or Work/Study Hours columns")

    return df_feat


def create_class12_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary flag for Class 12 students (high-risk group based on EDA).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Degree column or Degree_Class 12 one-hot column.

    Returns
    -------
    pd.DataFrame
        Dataset with Is_Class12 column added.
    """
    df_feat = df.copy()

    # Check for one-hot encoded column first
    if "Degree_Class 12" in df_feat.columns:
        df_feat["Is_Class12"] = df_feat["Degree_Class 12"].astype(int)
        print("✅ Created Is_Class12 flag from one-hot column")
    elif "Degree" in df_feat.columns:
        df_feat["Is_Class12"] = (df_feat["Degree"] == "Class 12").astype(int)
        print("✅ Created Is_Class12 flag from Degree column")
    else:
        print("⚠️ No Degree column found")

    return df_feat


# =============================================================================
# INTERACTION FEATURES
# =============================================================================


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between key variables.

    Features created:
    - AcademicPressure_x_Lifestyle: Academic Pressure × Lifestyle Score
    - FinancialStress_x_FamilyHistory: Financial Stress × Family History
    - AcademicPressure_x_Sleep: Academic Pressure × Sleep Encoded

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with required base features.

    Returns
    -------
    pd.DataFrame
        Dataset with interaction features added.
    """
    df_feat = df.copy()

    # Academic Pressure × Lifestyle
    if "Academic Pressure" in df_feat.columns and "Lifestyle_Score" in df_feat.columns:
        df_feat["AcademicPressure_x_Lifestyle"] = (
            df_feat["Academic Pressure"] * df_feat["Lifestyle_Score"]
        )
        print("✅ Created AcademicPressure_x_Lifestyle")

    # Financial Stress × Family History
    if (
        "Financial Stress" in df_feat.columns
        and "Family_History_Encoded" in df_feat.columns
    ):
        df_feat["FinancialStress_x_FamilyHistory"] = (
            df_feat["Financial Stress"] * df_feat["Family_History_Encoded"]
        )
        print("✅ Created FinancialStress_x_FamilyHistory")

    # Academic Pressure × Sleep
    if "Academic Pressure" in df_feat.columns and "Sleep_Encoded" in df_feat.columns:
        df_feat["AcademicPressure_x_Sleep"] = (
            df_feat["Academic Pressure"] * df_feat["Sleep_Encoded"]
        )
        print("✅ Created AcademicPressure_x_Sleep")

    # Total Stress × Lifestyle (stress-lifestyle interaction)
    if "Total_Stress" in df_feat.columns and "Lifestyle_Score" in df_feat.columns:
        df_feat["TotalStress_x_Lifestyle"] = (
            df_feat["Total_Stress"] * df_feat["Lifestyle_Score"]
        )
        print("✅ Created TotalStress_x_Lifestyle")

    return df_feat


# =============================================================================
# BINNING / DISCRETIZATION
# =============================================================================


def create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Age Groups from continuous Age variable.

    Groups:
    - Teen: 0-20
    - Young Adult: 21-25
    - Adult: 26-30
    - Mature: 31+

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Age column.

    Returns
    -------
    pd.DataFrame
        Dataset with Age_Group column added.
    """
    df_feat = df.copy()

    if "Age" in df_feat.columns:
        bins = [0, 20, 25, 30, 100]
        labels = ["Teen", "Young_Adult", "Adult", "Mature"]

        df_feat["Age_Group"] = pd.cut(
            df_feat["Age"], bins=bins, labels=labels, include_lowest=True
        )

        # Also create encoded version
        age_group_order = {label: i for i, label in enumerate(labels)}
        df_feat["Age_Group_Encoded"] = df_feat["Age_Group"].map(age_group_order)

        print(f"✅ Created Age_Group: {df_feat['Age_Group'].value_counts().to_dict()}")
    else:
        print("⚠️ Missing Age column")

    return df_feat


def create_cgpa_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create CGPA Categories from continuous CGPA variable.

    Categories:
    - Low: 0-6
    - Medium: 6-7.5
    - High: 7.5-9
    - Excellent: 9-10

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with CGPA column.

    Returns
    -------
    pd.DataFrame
        Dataset with CGPA_Category column added.
    """
    df_feat = df.copy()

    if "CGPA" in df_feat.columns:
        bins = [0, 6, 7.5, 9, 10]
        labels = ["Low", "Medium", "High", "Excellent"]

        df_feat["CGPA_Category"] = pd.cut(
            df_feat["CGPA"], bins=bins, labels=labels, include_lowest=True
        )

        # Also create encoded version
        cgpa_order = {label: i for i, label in enumerate(labels)}
        df_feat["CGPA_Category_Encoded"] = df_feat["CGPA_Category"].map(cgpa_order)

        print(
            f"✅ Created CGPA_Category: {df_feat['CGPA_Category'].value_counts().to_dict()}"
        )
    else:
        print("⚠️ Missing CGPA column")

    return df_feat


def create_stress_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create stress level categories from Academic and Financial stress.

    Levels:
    - Low: 1-2
    - Medium: 3
    - High: 4-5

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with stress columns.

    Returns
    -------
    pd.DataFrame
        Dataset with stress level columns added.
    """
    df_feat = df.copy()

    def categorize_stress(value):
        if pd.isna(value):
            return np.nan
        elif value <= 2:
            return "Low"
        elif value == 3:
            return "Medium"
        else:
            return "High"

    stress_order = {"Low": 0, "Medium": 1, "High": 2}

    # Academic Pressure Level
    if "Academic Pressure" in df_feat.columns:
        df_feat["Academic_Pressure_Level"] = df_feat["Academic Pressure"].apply(
            categorize_stress
        )
        df_feat["Academic_Pressure_Level_Encoded"] = df_feat[
            "Academic_Pressure_Level"
        ].map(stress_order)
        print("✅ Created Academic_Pressure_Level")

    # Financial Stress Level
    if "Financial Stress" in df_feat.columns:
        df_feat["Financial_Stress_Level"] = df_feat["Financial Stress"].apply(
            categorize_stress
        )
        df_feat["Financial_Stress_Level_Encoded"] = df_feat[
            "Financial_Stress_Level"
        ].map(stress_order)
        print("✅ Created Financial_Stress_Level")

    return df_feat


# =============================================================================
# RISK SCORE
# =============================================================================


def create_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a composite Risk Score based on known risk factors.

    Risk factors (from EDA insights):
    - Low Sleep (inverted: 3 - Sleep_Encoded)
    - Unhealthy Diet (inverted: 2 - Diet_Encoded)
    - High Financial Stress
    - High Academic Pressure
    - Family History of Mental Illness
    - Suicidal Thoughts
    - Is Class 12

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with encoded features.

    Returns
    -------
    pd.DataFrame
        Dataset with Risk_Score column added.
    """
    df_feat = df.copy()

    risk_score = pd.Series(0, index=df_feat.index, dtype=float)

    # Sleep risk (inverted: less sleep = higher risk)
    if "Sleep_Encoded" in df_feat.columns:
        max_sleep = df_feat["Sleep_Encoded"].max()
        risk_score += (
            (max_sleep - df_feat["Sleep_Encoded"]) / max_sleep * 2
        )  # Weight: 2

    # Diet risk (inverted: unhealthy = higher risk)
    if "Diet_Encoded" in df_feat.columns:
        max_diet = df_feat["Diet_Encoded"].max()
        risk_score += (max_diet - df_feat["Diet_Encoded"]) / max_diet * 2  # Weight: 2

    # Financial Stress (normalized)
    if "Financial Stress" in df_feat.columns:
        max_fin = df_feat["Financial Stress"].max()
        risk_score += (
            df_feat["Financial Stress"] / max_fin * 3
        )  # Weight: 3 (highest impact from EDA)

    # Academic Pressure (normalized)
    if "Academic Pressure" in df_feat.columns:
        max_acad = df_feat["Academic Pressure"].max()
        risk_score += df_feat["Academic Pressure"] / max_acad * 1.5  # Weight: 1.5

    # Family History
    if "Family_History_Encoded" in df_feat.columns:
        risk_score += df_feat["Family_History_Encoded"] * 1.5  # Weight: 1.5

    # Suicidal Thoughts
    if "Suicidal_Thoughts_Encoded" in df_feat.columns:
        risk_score += df_feat["Suicidal_Thoughts_Encoded"] * 2  # Weight: 2

    # Class 12 flag
    if "Is_Class12" in df_feat.columns:
        risk_score += df_feat["Is_Class12"] * 1  # Weight: 1

    df_feat["Risk_Score"] = risk_score
    print(
        f"✅ Created Risk_Score (range: {risk_score.min():.2f}-{risk_score.max():.2f})"
    )

    return df_feat


# =============================================================================
# FULL FEATURE ENGINEERING PIPELINE
# =============================================================================


def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with all engineered features.
    """
    print("=" * 60)
    print("🔧 FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # Step 1: Composite features
    print("\n📊 Step 1: Creating composite features...")
    df = create_lifestyle_score(df)
    df = create_total_stress(df)
    df = create_study_efficiency(df)
    df = create_class12_flag(df)

    # Step 2: Interaction features
    print("\n🔗 Step 2: Creating interaction features...")
    df = create_interaction_features(df)

    # Step 3: Binning/Discretization
    print("\n📏 Step 3: Creating categorical bins...")
    df = create_age_groups(df)
    df = create_cgpa_categories(df)
    df = create_stress_levels(df)

    # Step 4: Risk Score
    print("\n⚠️ Step 4: Creating risk score...")
    df = create_risk_score(df)

    print("\n" + "=" * 60)
    print(f"✅ FEATURE ENGINEERING COMPLETE")
    print(f"   Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("=" * 60)

    return df


# =============================================================================
# FEATURE SELECTION UTILITIES
# =============================================================================


def get_feature_columns(df: pd.DataFrame, exclude_target: bool = True) -> List[str]:
    """
    Get list of feature columns for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    exclude_target : bool
        Whether to exclude the Depression target column.

    Returns
    -------
    List[str]
        List of feature column names.
    """
    # Columns to exclude from features
    exclude_cols = ["Depression"] if exclude_target else []

    # Also exclude non-encoded categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    exclude_cols.extend(categorical_cols)

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return feature_cols


def prepare_modeling_data(df: pd.DataFrame, target_col: str = "Depression") -> tuple:
    """
    Prepare X and y for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features and target.
    target_col : str
        Name of target column.

    Returns
    -------
    tuple
        (X, y, feature_names)
    """
    feature_cols = get_feature_columns(df, exclude_target=True)

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Handle any remaining NaN values
    X = X.fillna(X.median())

    print(f"✅ Prepared modeling data:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   y distribution: {y.value_counts().to_dict()}")

    return X, y, feature_cols


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

    # Get project root
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "student_depression_dataset.csv")

    # Run preprocessing
    df_processed, _ = preprocess_pipeline(data_path)

    # Run feature engineering
    df_features = feature_engineering_pipeline(df_processed)

    print("\n📊 Final DataFrame Info:")
    print(df_features.info())

    print("\n📊 New features created:")
    new_features = [
        "Lifestyle_Score",
        "Total_Stress",
        "Study_Efficiency",
        "Is_Class12",
        "AcademicPressure_x_Lifestyle",
        "FinancialStress_x_FamilyHistory",
        "Age_Group",
        "CGPA_Category",
        "Risk_Score",
    ]
    for feat in new_features:
        if feat in df_features.columns:
            print(f"   ✓ {feat}")
