# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 📊 Student Depression Dataset - Preprocessing & Feature Engineering
#
# Notebook này thực hiện các bước tiền xử lý dữ liệu và tạo features mới cho việc dự đoán trầm cảm ở sinh viên.
#
# **Mục tiêu:**
# 1. Load và làm sạch dữ liệu
# 2. Handle missing values và invalid values
# 3. Encode categorical variables
# 4. Feature engineering (tạo features mới)
# 5. Export processed data

# %% [markdown]
# ## 1. Setup & Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "src"))
sys.path.insert(0, "../src")

warnings.filterwarnings("ignore")

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

# Plot settings
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

print("✅ Libraries imported successfully!")

# %% [markdown]
# ## 2. Load Raw Data

# %%
# Load the dataset
DATA_PATH = "../data/student_depression_dataset.csv"
df_raw = pd.read_csv(DATA_PATH)

print(f"📊 Dataset shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print(f"\n📋 Columns:")
for i, col in enumerate(df_raw.columns, 1):
    print(f"   {i:2}. {col}")

# %%
# Preview data
df_raw.head(10)

# %%
# Data types and info
df_raw.info()

# %% [markdown]
# ## 3. Data Cleaning

# %% [markdown]
# ### 3.1 Filter Students Only

# %%
# Check profession distribution
print("📊 Profession Distribution:")
print(df_raw["Profession"].value_counts())
print(f"\nStudent percentage: {(df_raw['Profession'] == 'Student').mean() * 100:.2f}%")

# %%
# Filter to students only (99.9% of data)
df = df_raw[df_raw["Profession"] == "Student"].copy()
print(f"✅ Filtered to students: {len(df)} rows")

# %% [markdown]
# ### 3.2 Drop Irrelevant Columns

# %%
# Columns to drop
drop_cols = ["id", "Work Pressure", "Job Satisfaction", "Profession", "City"]

# Check these columns before dropping
print("📋 Columns to drop:")
for col in drop_cols:
    if col in df.columns:
        unique_count = df[col].nunique()
        print(f"   • {col}: {unique_count} unique values")

# %%
# Drop columns
df = df.drop(columns=drop_cols, errors="ignore")
print(f"✅ Dropped {len(drop_cols)} columns. New shape: {df.shape}")

# %% [markdown]
# ### 3.3 Handle Invalid Values

# %%
# Check for invalid values in numeric columns
numeric_cols = [
    "Age",
    "Academic Pressure",
    "CGPA",
    "Study Satisfaction",
    "Work/Study Hours",
    "Financial Stress",
    "Depression",
]

print("🔍 Checking for invalid values in numeric columns:")
for col in numeric_cols:
    if col in df.columns:
        # Check for non-numeric values
        non_numeric = df[col].apply(
            lambda x: not isinstance(x, (int, float)) and pd.notna(x)
        )
        if non_numeric.any():
            invalid_vals = df.loc[non_numeric, col].unique()
            print(f"   ⚠️ {col}: {non_numeric.sum()} invalid values: {invalid_vals}")

# %%
# Clean invalid values - convert to numeric (coerce errors to NaN)
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("✅ Invalid numeric values converted to NaN")

# %%
# Check for invalid categorical values
print("\n🔍 Sleep Duration unique values:")
print(df["Sleep Duration"].value_counts())

# %%
# Handle 'Others' in Sleep Duration
others_count = (df["Sleep Duration"] == "Others").sum()
if others_count > 0:
    df.loc[df["Sleep Duration"] == "Others", "Sleep Duration"] = np.nan
    print(f"✅ Converted {others_count} 'Others' values to NaN in Sleep Duration")

# %%
# Check Dietary Habits
print("\n🔍 Dietary Habits unique values:")
print(df["Dietary Habits"].value_counts())

# %% [markdown]
# ### 3.4 Handle Missing Values

# %%
# Analyze missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame(
    {"Missing Count": missing, "Missing %": missing_pct}
).sort_values("Missing %", ascending=False)

missing_df[missing_df["Missing Count"] > 0]

# %%
# Visualize missing values
if missing_df["Missing Count"].sum() > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    missing_data = missing_df[missing_df["Missing Count"] > 0]
    ax.barh(missing_data.index, missing_data["Missing %"], color="coral")
    ax.set_xlabel("Missing %")
    ax.set_title("Missing Values by Column")
    plt.tight_layout()
    plt.show()
else:
    print("✅ No missing values to visualize!")

# %%
# Fill missing values
# Numerical: median
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numerical_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"   Filled {col} with median: {median_val}")

# Categorical: mode
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"   Filled {col} with mode: {mode_val}")

print(f"\n✅ Missing values handled. Total NaN remaining: {df.isnull().sum().sum()}")

# %% [markdown]
# ## 4. Encoding Categorical Variables

# %% [markdown]
# ### 4.1 Ordinal Encoding

# %%
# Sleep Duration - Ordinal encoding
sleep_order = {
    "'Less than 5 hours'": 0,
    "Less than 5 hours": 0,
    "'5-6 hours'": 1,
    "5-6 hours": 1,
    "'7-8 hours'": 2,
    "7-8 hours": 2,
    "'More than 8 hours'": 3,
    "More than 8 hours": 3,
}

df["Sleep_Encoded"] = df["Sleep Duration"].map(sleep_order)
print("✅ Encoded Sleep Duration:")
print("   0 = Less than 5 hours (worst)")
print("   1 = 5-6 hours")
print("   2 = 7-8 hours")
print("   3 = More than 8 hours (best)")
print(f"\n   Distribution: {df['Sleep_Encoded'].value_counts().sort_index().to_dict()}")

# %%
# Dietary Habits - Ordinal encoding
diet_order = {"Unhealthy": 0, "Moderate": 1, "Healthy": 2}

df["Diet_Encoded"] = df["Dietary Habits"].map(diet_order)
print("✅ Encoded Dietary Habits:")
print("   0 = Unhealthy")
print("   1 = Moderate")
print("   2 = Healthy")
print(f"\n   Distribution: {df['Diet_Encoded'].value_counts().sort_index().to_dict()}")

# %% [markdown]
# ### 4.2 Binary Encoding

# %%
# Gender
df["Gender_Encoded"] = (df["Gender"] == "Female").astype(int)
print(f"✅ Encoded Gender: 0=Male, 1=Female")
print(f"   Distribution: {df['Gender_Encoded'].value_counts().to_dict()}")

# %%
# Family History of Mental Illness
df["Family_History_Encoded"] = (df["Family History of Mental Illness"] == "Yes").astype(
    int
)
print(f"✅ Encoded Family History: 0=No, 1=Yes")
print(f"   Distribution: {df['Family_History_Encoded'].value_counts().to_dict()}")

# %%
# Suicidal Thoughts
suicidal_col = "Have you ever had suicidal thoughts ?"
df["Suicidal_Thoughts_Encoded"] = (df[suicidal_col] == "Yes").astype(int)
print(f"✅ Encoded Suicidal Thoughts: 0=No, 1=Yes")
print(f"   Distribution: {df['Suicidal_Thoughts_Encoded'].value_counts().to_dict()}")

# %% [markdown]
# ### 4.3 One-Hot Encoding for Degree

# %%
# Check Degree distribution
print("📊 Degree Distribution:")
print(df["Degree"].value_counts())

# %%
# One-hot encode Degree
degree_dummies = pd.get_dummies(df["Degree"], prefix="Degree")
df = pd.concat([df, degree_dummies], axis=1)
print(f"✅ One-hot encoded Degree: {len(degree_dummies.columns)} categories created")

# %% [markdown]
# ## 5. Feature Engineering

# %% [markdown]
# ### 5.1 Composite Features

# %%
# Lifestyle Score (Sleep + Diet combined)
df["Lifestyle_Score"] = df["Sleep_Encoded"] + df["Diet_Encoded"]
print(
    f"✅ Created Lifestyle_Score (range: {df['Lifestyle_Score'].min()}-{df['Lifestyle_Score'].max()})"
)
print("   Higher = healthier lifestyle")

# %%
# Total Stress Score
df["Total_Stress"] = df["Academic Pressure"] + df["Financial Stress"]
print(
    f"✅ Created Total_Stress (range: {df['Total_Stress'].min()}-{df['Total_Stress'].max()})"
)
print("   Higher = more stress")

# %%
# Study Efficiency
df["Study_Efficiency"] = df["CGPA"] / (df["Work/Study Hours"] + 1)
print(
    f"✅ Created Study_Efficiency (range: {df['Study_Efficiency'].min():.2f}-{df['Study_Efficiency'].max():.2f})"
)
print("   Higher = more efficient (high GPA with less hours)")

# %%
# Is Class 12 flag (high-risk group)
df["Is_Class12"] = (df["Degree"] == "Class 12").astype(int)
print(f"✅ Created Is_Class12 flag")
print(
    f"   Class 12 students: {df['Is_Class12'].sum()} ({df['Is_Class12'].mean()*100:.1f}%)"
)

# %% [markdown]
# ### 5.2 Interaction Features

# %%
# Academic Pressure × Lifestyle
df["AcademicPressure_x_Lifestyle"] = df["Academic Pressure"] * df["Lifestyle_Score"]
print("✅ Created AcademicPressure_x_Lifestyle")

# Financial Stress × Family History
df["FinancialStress_x_FamilyHistory"] = (
    df["Financial Stress"] * df["Family_History_Encoded"]
)
print("✅ Created FinancialStress_x_FamilyHistory")

# Total Stress × Lifestyle
df["TotalStress_x_Lifestyle"] = df["Total_Stress"] * df["Lifestyle_Score"]
print("✅ Created TotalStress_x_Lifestyle")

# %% [markdown]
# ### 5.3 Age & CGPA Categories

# %%
# Age Groups
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 20, 25, 30, 100],
    labels=["Teen", "Young_Adult", "Adult", "Mature"],
)
age_group_order = {"Teen": 0, "Young_Adult": 1, "Adult": 2, "Mature": 3}
df["Age_Group_Encoded"] = df["Age_Group"].map(age_group_order)
print(f"✅ Created Age_Group: {df['Age_Group'].value_counts().to_dict()}")

# %%
# CGPA Categories
df["CGPA_Category"] = pd.cut(
    df["CGPA"], bins=[0, 6, 7.5, 9, 10], labels=["Low", "Medium", "High", "Excellent"]
)
cgpa_order = {"Low": 0, "Medium": 1, "High": 2, "Excellent": 3}
df["CGPA_Category_Encoded"] = df["CGPA_Category"].map(cgpa_order)
print(f"✅ Created CGPA_Category: {df['CGPA_Category'].value_counts().to_dict()}")

# %% [markdown]
# ### 5.4 Stress Level Categories


# %%
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
df["Academic_Pressure_Level"] = df["Academic Pressure"].apply(categorize_stress)
df["Academic_Pressure_Level_Encoded"] = df["Academic_Pressure_Level"].map(stress_order)

# Financial Stress Level
df["Financial_Stress_Level"] = df["Financial Stress"].apply(categorize_stress)
df["Financial_Stress_Level_Encoded"] = df["Financial_Stress_Level"].map(stress_order)

print("✅ Created stress level categories (Low/Medium/High)")

# %% [markdown]
# ### 5.5 Risk Score

# %%
# Create composite Risk Score
risk_score = pd.Series(0, index=df.index, dtype=float)

# Sleep risk (inverted: less sleep = higher risk)
max_sleep = df["Sleep_Encoded"].max()
risk_score += (max_sleep - df["Sleep_Encoded"]) / max_sleep * 2  # Weight: 2

# Diet risk (inverted: unhealthy = higher risk)
max_diet = df["Diet_Encoded"].max()
risk_score += (max_diet - df["Diet_Encoded"]) / max_diet * 2  # Weight: 2

# Financial Stress (highest impact from EDA)
max_fin = df["Financial Stress"].max()
risk_score += df["Financial Stress"] / max_fin * 3  # Weight: 3

# Academic Pressure
max_acad = df["Academic Pressure"].max()
risk_score += df["Academic Pressure"] / max_acad * 1.5  # Weight: 1.5

# Family History
risk_score += df["Family_History_Encoded"] * 1.5  # Weight: 1.5

# Suicidal Thoughts
risk_score += df["Suicidal_Thoughts_Encoded"] * 2  # Weight: 2

# Class 12 flag
risk_score += df["Is_Class12"] * 1  # Weight: 1

df["Risk_Score"] = risk_score
print(f"✅ Created Risk_Score (range: {risk_score.min():.2f}-{risk_score.max():.2f})")

# %% [markdown]
# ## 6. Visualize Engineered Features

# %%
# Risk Score distribution by Depression status
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution
ax1 = axes[0]
df[df["Depression"] == 0]["Risk_Score"].hist(
    bins=30, alpha=0.7, label="No Depression", ax=ax1, color="green"
)
df[df["Depression"] == 1]["Risk_Score"].hist(
    bins=30, alpha=0.7, label="Depression", ax=ax1, color="red"
)
ax1.set_xlabel("Risk Score")
ax1.set_ylabel("Count")
ax1.set_title("Risk Score Distribution by Depression Status")
ax1.legend()

# Box plot
ax2 = axes[1]
df.boxplot(column="Risk_Score", by="Depression", ax=ax2)
ax2.set_xlabel("Depression (0=No, 1=Yes)")
ax2.set_ylabel("Risk Score")
ax2.set_title("Risk Score by Depression Status")
plt.suptitle("")

plt.tight_layout()
plt.show()

# %%
# Correlation of new features with Depression
new_features = [
    "Lifestyle_Score",
    "Total_Stress",
    "Study_Efficiency",
    "Is_Class12",
    "Risk_Score",
    "AcademicPressure_x_Lifestyle",
]

correlations = {}
for feat in new_features:
    if feat in df.columns:
        corr = df[feat].corr(df["Depression"])
        correlations[feat] = corr

corr_df = pd.DataFrame.from_dict(correlations, orient="index", columns=["Correlation"])
corr_df = corr_df.sort_values("Correlation", ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["red" if x > 0 else "green" for x in corr_df["Correlation"]]
ax.barh(corr_df.index, corr_df["Correlation"], color=colors)
ax.set_xlabel("Correlation with Depression")
ax.set_title("Engineered Features Correlation with Depression")
ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
plt.tight_layout()
plt.show()

print("\n📊 Feature Correlations with Depression:")
print(corr_df)

# %% [markdown]
# ## 7. Prepare Final Dataset

# %%
# Drop original categorical columns (keep only encoded versions)
cols_to_drop = [
    "Sleep Duration",
    "Dietary Habits",
    "Gender",
    "Family History of Mental Illness",
    "Have you ever had suicidal thoughts ?",
    "Degree",
    "Age_Group",
    "CGPA_Category",
    "Academic_Pressure_Level",
    "Financial_Stress_Level",
]

df_final = df.drop(
    columns=[c for c in cols_to_drop if c in df.columns], errors="ignore"
)
print(f"✅ Final dataset shape: {df_final.shape}")

# %%
# Display final columns
print("\n📋 Final Features:")
for i, col in enumerate(df_final.columns, 1):
    print(f"   {i:2}. {col}")

# %%
# Summary statistics
df_final.describe()

# %% [markdown]
# ## 8. Save Processed Data

# %%
# Save to CSV
output_path = "../results/processed_data_notebook.csv"
os.makedirs("../results", exist_ok=True)
df_final.to_csv(output_path, index=False)
print(f"✅ Saved processed data to: {output_path}")

# %%
# Also save as pickle for faster loading
df_final.to_pickle("../results/processed_data.pkl")
print("✅ Saved pickle file for faster loading")

# %% [markdown]
# ## 9. Summary
#
# ### Preprocessing Steps Completed:
# 1. ✅ Loaded 27,901 rows × 18 columns
# 2. ✅ Filtered to students only (27,870 rows)
# 3. ✅ Dropped irrelevant columns (id, Work Pressure, Job Satisfaction, etc.)
# 4. ✅ Cleaned invalid values ('?', 'Others')
# 5. ✅ Handled missing values (median/mode imputation)
# 6. ✅ Encoded categorical variables:
#    - Ordinal: Sleep Duration, Dietary Habits
#    - Binary: Gender, Family History, Suicidal Thoughts
#    - One-Hot: Degree
#
# ### Features Engineered:
# - `Lifestyle_Score`: Sleep + Diet (higher = healthier)
# - `Total_Stress`: Academic + Financial stress
# - `Study_Efficiency`: CGPA / Work Hours
# - `Is_Class12`: High-risk group flag
# - `Risk_Score`: Composite risk indicator
# - Interaction features: AcademicPressure_x_Lifestyle, etc.
# - Categorical bins: Age_Group, CGPA_Category, Stress_Level

# %%
print("🎉 Preprocessing complete! Ready for modeling.")
print(f"\n📊 Final dataset:")
print(f"   Rows: {len(df_final)}")
print(f"   Features: {len(df_final.columns) - 1}")  # -1 for target
print(
    f"   Target: Depression (0={sum(df_final['Depression']==0)}, 1={sum(df_final['Depression']==1)})"
)
