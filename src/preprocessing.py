"""
Student Depression Dataset - Preprocessing Module
================================================
Functions for data cleaning and preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# CONSTANTS
# =============================================================================

# Columns to drop (not relevant for student analysis)
DROP_COLUMNS = ["id", "Work Pressure", "Job Satisfaction", "Profession", "City"]

# Ordinal encoding mappings
SLEEP_DURATION_ORDER = {
    "'Less than 5 hours'": 0,
    "Less than 5 hours": 0,
    "'5-6 hours'": 1,
    "5-6 hours": 1,
    "'7-8 hours'": 2,
    "7-8 hours": 2,
    "'More than 8 hours'": 3,
    "More than 8 hours": 3,
}

DIETARY_HABITS_ORDER = {
    "Unhealthy": 0,
    "Moderate": 1,
    "Healthy": 2,
}

BINARY_YES_NO = {
    "No": 0,
    "Yes": 1,
}

GENDER_MAPPING = {
    "Male": 0,
    "Female": 1,
}


# =============================================================================
# DATA LOADING
# =============================================================================


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the student depression dataset from CSV.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# =============================================================================
# DATA FILTERING
# =============================================================================


def filter_students(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to keep only students (99.9% of data).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Filtered dataset with only students.
    """
    initial_count = len(df)
    df_students = df[df["Profession"] == "Student"].copy()
    filtered_count = len(df_students)

    print(
        f"✅ Filtered to students: {filtered_count}/{initial_count} rows ({filtered_count/initial_count*100:.1f}%)"
    )
    return df_students


def drop_irrelevant_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Drop columns that are not relevant for student analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    columns : list, optional
        List of column names to drop. If None, uses default DROP_COLUMNS.

    Returns
    -------
    pd.DataFrame
        Dataset with columns dropped.
    """
    if columns is None:
        columns = DROP_COLUMNS

    cols_to_drop = [col for col in columns if col in df.columns]
    df_dropped = df.drop(columns=cols_to_drop, errors="ignore")

    print(f"✅ Dropped columns: {cols_to_drop}")
    return df_dropped


# =============================================================================
# DATA CLEANING - INVALID VALUES
# =============================================================================


def clean_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean invalid values in the dataset (e.g., '?' or 'Others' in numeric/categorical columns).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to clean.

    Returns
    -------
    pd.DataFrame
        Dataset with invalid values converted to NaN.
    """
    df_clean = df.copy()

    # Define columns that should be numeric
    numeric_cols = [
        "Age",
        "Academic Pressure",
        "CGPA",
        "Study Satisfaction",
        "Work/Study Hours",
        "Financial Stress",
        "Depression",
    ]

    invalid_count = 0

    for col in numeric_cols:
        if col in df_clean.columns:
            # Convert to numeric, coercing errors to NaN
            original_nulls = df_clean[col].isnull().sum()
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            new_nulls = df_clean[col].isnull().sum()

            if new_nulls > original_nulls:
                invalid_count += new_nulls - original_nulls
                print(
                    f"   ⚠️ {col}: {new_nulls - original_nulls} invalid values converted to NaN"
                )

    # Handle 'Others' in Sleep Duration and Dietary Habits
    if "Sleep Duration" in df_clean.columns:
        others_count = (df_clean["Sleep Duration"] == "Others").sum()
        if others_count > 0:
            df_clean.loc[df_clean["Sleep Duration"] == "Others", "Sleep Duration"] = (
                np.nan
            )
            invalid_count += others_count
            print(
                f"   ⚠️ Sleep Duration: {others_count} 'Others' values converted to NaN"
            )

    if "Dietary Habits" in df_clean.columns:
        # Check for any unknown values
        known_values = ["Unhealthy", "Moderate", "Healthy"]
        unknown_mask = (
            ~df_clean["Dietary Habits"].isin(known_values)
            & df_clean["Dietary Habits"].notna()
        )
        unknown_count = unknown_mask.sum()
        if unknown_count > 0:
            df_clean.loc[unknown_mask, "Dietary Habits"] = np.nan
            invalid_count += unknown_count
            print(
                f"   ⚠️ Dietary Habits: {unknown_count} unknown values converted to NaN"
            )

    if invalid_count > 0:
        print(f"✅ Cleaned {invalid_count} invalid values total")
    else:
        print("✅ No invalid values found")

    return df_clean


# =============================================================================
# MISSING VALUE HANDLING
# =============================================================================


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.

    Returns
    -------
    pd.DataFrame
        Summary of missing values per column.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(
        "Missing %", ascending=False
    )

    if len(missing_df) == 0:
        print("✅ No missing values found!")
    else:
        print(f"⚠️ Found missing values in {len(missing_df)} columns:")
        print(missing_df)

    return missing_df


def handle_missing_values(
    df: pd.DataFrame,
    numerical_strategy: str = "median",
    categorical_strategy: str = "mode",
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with missing values.
    numerical_strategy : str
        Strategy for numerical columns: 'median', 'mean', or 'drop'.
    categorical_strategy : str
        Strategy for categorical columns: 'mode' or 'drop'.

    Returns
    -------
    pd.DataFrame
        Dataset with missing values handled.
    """
    df_clean = df.copy()

    # Identify column types
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()

    # Handle numerical columns
    for col in numerical_cols:
        if df_clean[col].isnull().any():
            if numerical_strategy == "median":
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif numerical_strategy == "mean":
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif numerical_strategy == "drop":
                df_clean.dropna(subset=[col], inplace=True)

    # Handle categorical columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            if categorical_strategy == "mode":
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col].fillna(mode_val[0], inplace=True)
            elif categorical_strategy == "drop":
                df_clean.dropna(subset=[col], inplace=True)

    print(f"✅ Missing values handled: {len(df_clean)} rows remaining")
    return df_clean


# =============================================================================
# ENCODING
# =============================================================================


def encode_sleep_duration(
    df: pd.DataFrame, column: str = "Sleep Duration"
) -> pd.DataFrame:
    """
    Encode Sleep Duration as ordinal variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    column : str
        Name of the sleep duration column.

    Returns
    -------
    pd.DataFrame
        Dataset with encoded sleep duration.
    """
    df_encoded = df.copy()

    if column in df_encoded.columns:
        df_encoded["Sleep_Encoded"] = df_encoded[column].map(SLEEP_DURATION_ORDER)

        # Check for unmapped values
        unmapped = df_encoded["Sleep_Encoded"].isnull().sum()
        if unmapped > 0:
            print(f"⚠️ {unmapped} unmapped values in {column}")
            unique_vals = df_encoded[df_encoded["Sleep_Encoded"].isnull()][
                column
            ].unique()
            print(f"   Unmapped values: {unique_vals}")
        else:
            print(f"✅ Encoded {column}: 0=<5h, 1=5-6h, 2=7-8h, 3=>8h")

    return df_encoded


def encode_dietary_habits(
    df: pd.DataFrame, column: str = "Dietary Habits"
) -> pd.DataFrame:
    """
    Encode Dietary Habits as ordinal variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    column : str
        Name of the dietary habits column.

    Returns
    -------
    pd.DataFrame
        Dataset with encoded dietary habits.
    """
    df_encoded = df.copy()

    if column in df_encoded.columns:
        df_encoded["Diet_Encoded"] = df_encoded[column].map(DIETARY_HABITS_ORDER)

        unmapped = df_encoded["Diet_Encoded"].isnull().sum()
        if unmapped > 0:
            print(f"⚠️ {unmapped} unmapped values in {column}")
        else:
            print(f"✅ Encoded {column}: 0=Unhealthy, 1=Moderate, 2=Healthy")

    return df_encoded


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary columns (Yes/No and Gender).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with binary columns encoded.
    """
    df_encoded = df.copy()

    # Gender
    if "Gender" in df_encoded.columns:
        df_encoded["Gender_Encoded"] = df_encoded["Gender"].map(GENDER_MAPPING)
        print(f"✅ Encoded Gender: 0=Male, 1=Female")

    # Family History
    if "Family History of Mental Illness" in df_encoded.columns:
        df_encoded["Family_History_Encoded"] = df_encoded[
            "Family History of Mental Illness"
        ].map(BINARY_YES_NO)
        print(f"✅ Encoded Family History: 0=No, 1=Yes")

    # Suicidal Thoughts
    suicidal_col = "Have you ever had suicidal thoughts ?"
    if suicidal_col in df_encoded.columns:
        df_encoded["Suicidal_Thoughts_Encoded"] = df_encoded[suicidal_col].map(
            BINARY_YES_NO
        )
        print(f"✅ Encoded Suicidal Thoughts: 0=No, 1=Yes")

    return df_encoded


def encode_degree_onehot(df: pd.DataFrame, column: str = "Degree") -> pd.DataFrame:
    """
    One-hot encode the Degree column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    column : str
        Name of the degree column.

    Returns
    -------
    pd.DataFrame
        Dataset with one-hot encoded degree.
    """
    df_encoded = df.copy()

    if column in df_encoded.columns:
        degree_dummies = pd.get_dummies(df_encoded[column], prefix="Degree")
        df_encoded = pd.concat([df_encoded, degree_dummies], axis=1)
        print(f"✅ One-hot encoded {column}: {len(degree_dummies.columns)} categories")

    return df_encoded


# =============================================================================
# FULL PREPROCESSING PIPELINE
# =============================================================================


def preprocess_pipeline(
    filepath: str, drop_original_categorical: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    drop_original_categorical : bool
        Whether to drop original categorical columns after encoding.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (preprocessed_df, original_df)
    """
    print("=" * 60)
    print("🚀 PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    print("\n📥 Step 1: Loading data...")
    df_original = load_data(filepath)

    # Step 2: Filter students
    print("\n🎓 Step 2: Filtering students...")
    df = filter_students(df_original)

    # Step 3: Drop irrelevant columns
    print("\n🗑️ Step 3: Dropping irrelevant columns...")
    df = drop_irrelevant_columns(df)

    # Step 4: Clean invalid values (e.g., '?', 'Others')
    print("\n🧹 Step 4: Cleaning invalid values...")
    df = clean_invalid_values(df)

    # Step 5: Analyze and handle missing values
    print("\n🔍 Step 5: Handling missing values...")
    analyze_missing_values(df)
    df = handle_missing_values(df)

    # Step 6: Encode categorical variables
    print("\n🔢 Step 6: Encoding categorical variables...")
    df = encode_sleep_duration(df)
    df = encode_dietary_habits(df)
    df = encode_binary_columns(df)
    df = encode_degree_onehot(df)

    # Step 7: Drop original categorical columns if requested
    if drop_original_categorical:
        cat_cols_to_drop = [
            "Sleep Duration",
            "Dietary Habits",
            "Gender",
            "Family History of Mental Illness",
            "Have you ever had suicidal thoughts ?",
            "Degree",
        ]
        df = df.drop(
            columns=[c for c in cat_cols_to_drop if c in df.columns], errors="ignore"
        )
        print(f"\n🗑️ Dropped original categorical columns")

    print("\n" + "=" * 60)
    print(f"✅ PREPROCESSING COMPLETE")
    print(f"   Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("=" * 60)

    return df, df_original


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    import os

    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "student_depression_dataset.csv")

    # Run pipeline
    df_processed, df_original = preprocess_pipeline(data_path)

    print("\n📊 Processed DataFrame Info:")
    print(df_processed.info())
    print("\n📊 Sample rows:")
    print(df_processed.head())
