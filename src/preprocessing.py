import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional

SLEEP_DURATION_MAPPING = {
    'Less than 5 hours': 1,
    '5-6 hours': 2,
    '7-8 hours': 3,
    'More than 8 hours': 4
}

DIETARY_HABITS_MAPPING = {
    'Unhealthy': 1,
    'Moderate': 2,
    'Healthy': 3
}

BINARY_YES_NO_MAPPING = {
    'Yes': 1,
    'No': 0
}

GENDER_MAPPING = {
    'Male': 0,
    'Female': 1
}


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def get_data_overview(df: pd.DataFrame) -> Dict:
    overview = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
    }
    return overview


def get_missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_data = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(
        'Missing %', ascending=False
    )
    return missing_data


def get_duplicates_count(df: pd.DataFrame) -> int:
    return df.duplicated().sum()


def clean_sleep_duration(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    if 'Sleep Duration' in df_clean.columns:
        df_clean['Sleep Duration'] = df_clean['Sleep Duration'].str.replace("'", "")
    return df_clean


def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    if 'id' in df_clean.columns:
        df_clean = df_clean.drop('id', axis=1)
    return df_clean


def encode_sleep_duration(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()
    if 'Sleep Duration' in df_encoded.columns:
        df_encoded['Sleep_Hours_Encoded'] = df_encoded['Sleep Duration'].map(SLEEP_DURATION_MAPPING)
    return df_encoded


def encode_dietary_habits(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()
    if 'Dietary Habits' in df_encoded.columns:
        df_encoded['Diet_Encoded'] = df_encoded['Dietary Habits'].map(DIETARY_HABITS_MAPPING)
    return df_encoded


def encode_suicidal_thoughts(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()
    col_name = 'Have you ever had suicidal thoughts ?'
    if col_name in df_encoded.columns:
        df_encoded['Suicidal_Thoughts'] = df_encoded[col_name].map(BINARY_YES_NO_MAPPING)
    return df_encoded


def encode_family_history(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()
    col_name = 'Family History of Mental Illness'
    if col_name in df_encoded.columns:
        df_encoded['Family_History'] = df_encoded[col_name].map(BINARY_YES_NO_MAPPING)
    return df_encoded


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()
    if 'Gender' in df_encoded.columns:
        df_encoded['Gender_Encoded'] = df_encoded['Gender'].map(GENDER_MAPPING)
    return df_encoded


def encode_all_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()
    df_encoded = encode_sleep_duration(df_encoded)
    df_encoded = encode_dietary_habits(df_encoded)
    df_encoded = encode_suicidal_thoughts(df_encoded)
    df_encoded = encode_family_history(df_encoded)
    df_encoded = encode_gender(df_encoded)
    return df_encoded


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean = clean_sleep_duration(df_clean)
    df_clean = drop_id_column(df_clean)
    df_clean = encode_all_categorical(df_clean)
    return df_clean


def get_variable_classification() -> Dict[str, List[str]]:
    numerical_vars = [
        'Age', 'CGPA', 'Work/Study Hours', 'Academic Pressure',
        'Work Pressure', 'Study Satisfaction', 'Job Satisfaction',
        'Financial Stress'
    ]

    categorical_vars = [
        'Gender', 'City', 'Profession', 'Sleep Duration',
        'Dietary Habits', 'Degree',
        'Have you ever had suicidal thoughts ?',
        'Family History of Mental Illness'
    ]

    target_var = 'Depression'

    return {
        'numerical': numerical_vars,
        'categorical': categorical_vars,
        'target': target_var
    }


def get_column_descriptions() -> Dict[str, Dict[str, str]]:
    descriptions = {
        'id': {
            'type': 'Numerical (ID)',
            'description': 'Unique identifier for each student',
            'values': 'Positive integer'
        },
        'Gender': {
            'type': 'Categorical',
            'description': 'Gender of the student',
            'values': 'Male, Female'
        },
        'Age': {
            'type': 'Numerical (Continuous)',
            'description': 'Age of the student',
            'values': 'Integer (18-35+)'
        },
        'City': {
            'type': 'Categorical',
            'description': 'City of residence/study',
            'values': 'Cities in India'
        },
        'Profession': {
            'type': 'Categorical',
            'description': 'Profession (mainly Student in this dataset)',
            'values': 'Student, Working Professional...'
        },
        'Academic Pressure': {
            'type': 'Numerical (Ordinal)',
            'description': 'Level of academic pressure (self-assessed)',
            'values': '1-5 (1: Lowest, 5: Highest)'
        },
        'Work Pressure': {
            'type': 'Numerical (Ordinal)',
            'description': 'Level of work pressure',
            'values': '0-5 (0 if not working)'
        },
        'CGPA': {
            'type': 'Numerical (Continuous)',
            'description': 'Cumulative Grade Point Average',
            'values': '0.0-10.0'
        },
        'Study Satisfaction': {
            'type': 'Numerical (Ordinal)',
            'description': 'Satisfaction with studies',
            'values': '1-5 (1: Very unsatisfied, 5: Very satisfied)'
        },
        'Job Satisfaction': {
            'type': 'Numerical (Ordinal)',
            'description': 'Job satisfaction level',
            'values': '0-5 (0 if no job)'
        },
        'Sleep Duration': {
            'type': 'Categorical (Ordinal)',
            'description': 'Average daily sleep duration',
            'values': "'Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'"
        },
        'Dietary Habits': {
            'type': 'Categorical (Ordinal)',
            'description': 'Eating habits',
            'values': 'Healthy, Moderate, Unhealthy'
        },
        'Degree': {
            'type': 'Categorical',
            'description': 'Degree being pursued',
            'values': 'Class 12, BA, BSc, BE, B.Com, MBA, PhD, ...'
        },
        'Have you ever had suicidal thoughts ?': {
            'type': 'Categorical (Binary)',
            'description': 'History of suicidal thoughts',
            'values': 'Yes, No'
        },
        'Work/Study Hours': {
            'type': 'Numerical (Continuous)',
            'description': 'Daily work/study hours',
            'values': '0-12+'
        },
        'Financial Stress': {
            'type': 'Numerical (Ordinal)',
            'description': 'Level of financial stress',
            'values': '1-5 (1: Lowest, 5: Highest)'
        },
        'Family History of Mental Illness': {
            'type': 'Categorical (Binary)',
            'description': 'Family history of mental illness',
            'values': 'Yes, No'
        },
        'Depression': {
            'type': 'Categorical (Binary) - TARGET',
            'description': 'TARGET VARIABLE: Depression status',
            'values': '0: No depression, 1: Has depression'
        }
    }
    return descriptions


def preprocess_pipeline(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_original = load_data(data_path)
    df_processed = preprocess_data(df_original)
    return df_processed, df_original
