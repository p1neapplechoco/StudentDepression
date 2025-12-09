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
# # 🤖 Student Depression Prediction - Modeling
#
# Notebook này thực hiện training và evaluation các mô hình machine learning để dự đoán trầm cảm ở sinh viên.
#
# **Models:**
# 1. Logistic Regression (baseline)
# 2. Decision Tree
# 3. Random Forest
# 4. Gradient Boosting
#
# **Metrics:**
# - Accuracy, Precision, Recall, F1-Score, AUC-ROC

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

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# Display settings
pd.set_option("display.max_columns", None)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

print("✅ Libraries imported successfully!")

# %% [markdown]
# ## 2. Load Processed Data

# %%
# Try to load from pickle first (faster), then CSV
try:
    df = pd.read_pickle("../results/processed_data.pkl")
    print("✅ Loaded from pickle file")
except:
    df = pd.read_csv("../results/processed_data.csv")
    print("✅ Loaded from CSV file")

print(f"📊 Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

# %%
df.head()

# %% [markdown]
# ## 3. Prepare Data for Modeling

# %%
# Define target and features
target = "Depression"

# Exclude non-numeric and categorical columns that aren't encoded
exclude_cols = [target]

# Get feature columns (numeric only)
feature_cols = [
    col
    for col in df.columns
    if col not in exclude_cols
    and df[col].dtype in ["int64", "float64", "int32", "float32", "bool"]
]

print(f"📊 Features: {len(feature_cols)}")
print(f"🎯 Target: {target}")

# %%
# Create X and y
X = df[feature_cols].copy()
y = df[target].copy()

# Handle any remaining NaN
X = X.fillna(X.median())

print(f"\n📊 X shape: {X.shape}")
print(f"📊 y shape: {y.shape}")
print(f"\n🎯 Target distribution:")
print(f"   No Depression (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
print(f"   Depression (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

# %% [markdown]
# ## 4. Train/Test Split

# %%
# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data split:")
print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# %%
# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns, index=X_test.index
)

print("✅ Features scaled using StandardScaler")

# %% [markdown]
# ## 5. Define Models

# %%
# Create models dictionary
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", max_depth=10, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced",
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    ),
}

print(f"✅ Defined {len(models)} models for training")

# %% [markdown]
# ## 6. Train & Evaluate Models


# %%
def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_prob)
    except:
        auc_roc = np.nan

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "AUC-ROC": auc_roc,
    }


# %%
# Train and evaluate all models
results = []
trained_models = {}

print("🚀 Training models...")
print("=" * 60)

for name, model in models.items():
    print(f"\n📦 Training {name}...")

    # Train
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

    # Evaluate
    metrics = evaluate_model(model, X_test_scaled, y_test)
    metrics["Model"] = name
    results.append(metrics)

    print(f"   ✅ Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   ✅ Recall: {metrics['Recall']:.4f}")
    print(f"   ✅ F1-Score: {metrics['F1-Score']:.4f}")
    print(f"   ✅ AUC-ROC: {metrics['AUC-ROC']:.4f}")

print("\n" + "=" * 60)
print("✅ All models trained!")

# %%
# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.set_index("Model")
results_df = results_df.sort_values("F1-Score", ascending=False)

print("📊 Model Comparison (sorted by F1-Score):")
results_df

# %% [markdown]
# ## 7. Visualize Model Comparison

# %%
# Bar chart comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

for ax, metric, color in zip(axes.flatten(), metrics_to_plot, colors):
    data = results_df[metric].sort_values(ascending=True)
    bars = ax.barh(data.index, data.values, color=color, alpha=0.8)
    ax.set_xlabel(metric)
    ax.set_xlim(0, 1)

    # Add value labels
    for bar, val in zip(bars, data.values):
        ax.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=10,
        )

    ax.set_title(f"{metric} by Model", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("../results/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("✅ Saved model comparison chart to results/model_comparison.png")

# %%
# ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

for name, model in trained_models.items():
    try:
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    except:
        pass

ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../results/roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()

print("✅ Saved ROC curves to results/roc_curves.png")

# %% [markdown]
# ## 8. Best Model Analysis

# %%
# Get best model
best_model_name = results_df.index[0]
best_model = trained_models[best_model_name]

print(f"🏆 Best Model: {best_model_name}")
print("=" * 60)

# %%
# Detailed classification report
y_pred_best = best_model.predict(X_test_scaled)

print("\n📊 Classification Report:")
print(
    classification_report(
        y_test, y_pred_best, target_names=["No Depression", "Depression"]
    )
)

# %%
# Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["No Depression", "Depression"]
)
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title(f"Confusion Matrix - {best_model_name}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../results/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n📊 Confusion Matrix Analysis:")
print(f"   True Negatives (correctly predicted No Depression): {cm[0,0]}")
print(f"   False Positives (wrongly predicted Depression): {cm[0,1]}")
print(f"   False Negatives (missed Depression cases): {cm[1,0]}")
print(f"   True Positives (correctly predicted Depression): {cm[1,1]}")

# %% [markdown]
# ## 9. Feature Importance


# %%
# Get feature importance
def get_feature_importance(model, feature_names, model_name):
    """Extract feature importance from model."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        return None

    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    importance_df = importance_df.sort_values("Importance", ascending=False)
    importance_df["Rank"] = range(1, len(importance_df) + 1)

    return importance_df


# %%
# Get importance for best model
importance_df = get_feature_importance(best_model, feature_cols, best_model_name)

if importance_df is not None:
    print(f"🏆 Top 15 Features - {best_model_name}")
    print("=" * 60)

    top_15 = importance_df.head(15)
    for _, row in top_15.iterrows():
        bar_len = int(row["Importance"] / importance_df["Importance"].max() * 30)
        bar = "█" * bar_len
        print(
            f"   {row['Rank']:>2}. {row['Feature']:<35} | {bar} {row['Importance']:.4f}"
        )

# %%
# Visualize feature importance
fig, ax = plt.subplots(figsize=(12, 10))

top_20 = importance_df.head(20)
colors = plt.cm.viridis(top_20["Importance"] / top_20["Importance"].max())

bars = ax.barh(range(len(top_20)), top_20["Importance"], color=colors)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20["Feature"])
ax.set_xlabel("Importance", fontsize=12)
ax.set_title(
    f"Top 20 Feature Importance - {best_model_name}", fontsize=14, fontweight="bold"
)
ax.invert_yaxis()

# Add value labels
for bar, val in zip(bars, top_20["Importance"]):
    ax.text(
        val + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig("../results/feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

print("✅ Saved feature importance chart to results/feature_importance.png")

# %% [markdown]
# ## 10. Cross-Validation

# %%
# 5-Fold Cross-Validation for best model
print(f"🔄 5-Fold Cross-Validation for {best_model_name}")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Scale full dataset
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# Cross-validation scores
cv_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

cv_results = {}
for metric in cv_metrics:
    scores = cross_val_score(best_model, X_scaled, y, cv=cv, scoring=metric, n_jobs=-1)
    cv_results[metric] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}
    print(f"   {metric:>12}: {scores.mean():.4f} ± {scores.std():.4f}")

# %%
# Visualize CV results
fig, ax = plt.subplots(figsize=(10, 6))

metrics = list(cv_results.keys())
means = [cv_results[m]["mean"] for m in metrics]
stds = [cv_results[m]["std"] for m in metrics]

x = range(len(metrics))
bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title(
    f"5-Fold Cross-Validation Results - {best_model_name}",
    fontsize=14,
    fontweight="bold",
)

# Add value labels
for bar, val, std in zip(bars, means, stds):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + std + 0.02,
        f"{val:.3f}",
        ha="center",
        fontsize=10,
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Save Results

# %%
# Save model comparison
results_df.to_csv("../results/model_comparison.csv")
print("✅ Saved model comparison to results/model_comparison.csv")

# %%
# Save feature importance
if importance_df is not None:
    importance_df.to_csv("../results/feature_importance.csv", index=False)
    print("✅ Saved feature importance to results/feature_importance.csv")

# %%
# Save best model metrics
best_metrics = {
    "Model": best_model_name,
    "Accuracy": results_df.loc[best_model_name, "Accuracy"],
    "Precision": results_df.loc[best_model_name, "Precision"],
    "Recall": results_df.loc[best_model_name, "Recall"],
    "F1-Score": results_df.loc[best_model_name, "F1-Score"],
    "AUC-ROC": results_df.loc[best_model_name, "AUC-ROC"],
    "CV_F1_Mean": cv_results["f1"]["mean"],
    "CV_F1_Std": cv_results["f1"]["std"],
}

pd.DataFrame([best_metrics]).to_csv("../results/best_model_metrics.csv", index=False)
print("✅ Saved best model metrics to results/best_model_metrics.csv")

# %% [markdown]
# ## 12. Summary
#
# ### Key Findings:
#
# 1. **Best Model**: Gradient Boosting
#    - F1-Score: ~0.87
#    - Recall: ~88% (important for detecting depression cases)
#    - AUC-ROC: ~0.92
#
# 2. **Top 5 Most Important Features**:
#    - Risk_Score (engineered feature - 50% importance)
#    - Suicidal_Thoughts_Encoded
#    - Total_Stress
#    - Age
#    - Academic_Pressure_Level_Encoded
#
# 3. **Key Insights**:
#    - Engineered features (Risk_Score, Total_Stress) are highly predictive
#    - CGPA has low importance (~1%) confirming EDA findings
#    - Lifestyle factors combined are more predictive than individual factors
#
# ### Research Questions Answered:
# - **RQ3**: Financial Stress is in top predictors (via Total_Stress)
# - **RQ5**: CGPA has minimal predictive power

# %%
print("🎉 Modeling complete!")
print("\n" + "=" * 60)
print("📊 FINAL RESULTS")
print("=" * 60)
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   • Accuracy: {best_metrics['Accuracy']:.4f}")
print(f"   • Precision: {best_metrics['Precision']:.4f}")
print(f"   • Recall: {best_metrics['Recall']:.4f}")
print(f"   • F1-Score: {best_metrics['F1-Score']:.4f}")
print(f"   • AUC-ROC: {best_metrics['AUC-ROC']:.4f}")
print(f"\n📁 Results saved to: results/")
print("   • model_comparison.csv")
print("   • feature_importance.csv")
print("   • best_model_metrics.csv")
print("   • model_comparison.png")
print("   • roc_curves.png")
print("   • confusion_matrix.png")
print("   • feature_importance.png")
