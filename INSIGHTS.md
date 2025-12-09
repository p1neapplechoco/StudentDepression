
# 📊 Student Depression Dataset - Phân tích & Kế hoạch Nghiên cứu

> **Dataset**: 27,901 bản ghi × 18 features  
> **Target Variable**: `Depression` (0/1 - Binary Classification)

---

## 1. 🔍 Các Insights Nổi Bật (Key Findings)

### 🚨 "Điểm nóng" Lớp 12 (Class 12)
| Insight | Chi tiết |
|---------|----------|
| **Dữ liệu** | Học sinh **Class 12** có tỷ lệ trầm cảm **70.8%** (cao nhất) |
| **So sánh** | Cao hơn trung bình (58.5%), PhD, MBBS |
| **Ý nghĩa** | Áp lực thi cử/chuyển cấp > Áp lực học thuật đại học |

### 💤 Giấc ngủ & Ăn uống: "Combo" nguy hiểm
| Factor | Tỷ lệ trầm cảm |
|--------|----------------|
| Ngủ < 5 tiếng/ngày | **64.5%** |
| Ăn uống Unhealthy | **70.7%** |
| Ăn uống Healthy | 45.4% |

### 💸 Áp lực Tài chính (Financial Stress)
- **Financial Stress = 5/5** → Tỷ lệ trầm cảm **81.3%**
- Đây là yếu tố "ẩn" nhưng có sức tàn phá lớn nhất

### 🎓 Nghịch lý Điểm số (CGPA Paradox)
| Nhóm | CGPA trung bình |
|------|-----------------|
| Trầm cảm | 7.68 |
| Không trầm cảm | 7.62 |

→ **High-functioning depression** có thể đang hiện hữu

---

## 2. 🎯 Research Questions (Câu hỏi Nghiên cứu)

### RQ1: Hiệu ứng "Transition Stress"
> **"Áp lực giai đoạn chuyển tiếp (Class 12 → Đại học) có phải là yếu tố gây trầm cảm mạnh hơn áp lực học thuật thông thường?"**

**Hypothesis (H1)**: Học sinh Class 12 có Academic Pressure cao hơn trung bình, nhưng yếu tố Sleep và Dietary Habits của họ tệ hơn đáng kể.

**Kiểm chứng**: So sánh phân phối các features giữa nhóm Class 12 vs. các Degree khác.

---

### RQ2: Lifestyle Buffer Effect
> **"Lối sống lành mạnh (Ngủ đủ + Ăn healthy) có thể làm giảm tác động tiêu cực của Academic Pressure lên trầm cảm không?"**

**Hypothesis (H2)**: Trong nhóm có Academic Pressure cao (4-5), những người có Healthy Lifestyle sẽ có tỷ lệ trầm cảm thấp hơn đáng kể so với nhóm Unhealthy Lifestyle.

**Kiểm chứng**: Phân tích interaction effect giữa `Academic Pressure` × `Lifestyle Score` (Sleep + Diet combined).

---

### RQ3: Financial Stress as Hidden Killer
> **"Financial Stress có phải là yếu tố dự báo trầm cảm mạnh nhất, vượt trội hơn cả Academic Pressure?"**

**Hypothesis (H3)**: Trong mô hình dự đoán, `Financial Stress` sẽ có feature importance cao nhất.

**Kiểm chứng**: So sánh feature importance từ nhiều mô hình khác nhau.

---

### RQ4: Risk Profile Clustering
> **"Có thể phân nhóm sinh viên thành các 'Risk Profiles' dựa trên đặc điểm của họ không?"**

**Hypothesis (H4)**: Tồn tại ít nhất 3-4 clusters với tỷ lệ trầm cảm khác biệt rõ rệt (ví dụ: "High Risk", "Moderate Risk", "Low Risk").

**Kiểm chứng**: K-Means/Hierarchical Clustering → So sánh Depression rate giữa các clusters.

---

### RQ5: The CGPA Paradox
> **"Tại sao CGPA không khác biệt giữa nhóm trầm cảm và không trầm cảm?"**

**Hypothesis (H5)**: "High-functioning depression" - Sinh viên trầm cảm có thể đang "over-compensate" bằng cách học nhiều hơn, dẫn đến CGPA cao nhưng hy sinh sức khỏe tinh thần.

**Kiểm chứng**: Phân tích correlation giữa `Work/Study Hours`, `CGPA`, và `Depression` trong từng subgroup.

---

### RQ6: Family History as Genetic/Environmental Factor
> **"Tiền sử gia đình về bệnh tâm thần có phải là yếu tố làm tăng 'vulnerability' đối với các stressor khác không?"**

**Hypothesis (H6)**: Ở nhóm có `Family History = Yes`, mối quan hệ giữa `Financial Stress` → `Depression` sẽ mạnh hơn so với nhóm `Family History = No`.

**Kiểm chứng**: Moderation analysis hoặc stratified analysis.

---

## 3. 📋 Kế hoạch Preprocessing & Modeling

### Phase 1: Data Cleaning & Preprocessing

#### 1.1 Filter Data
```python
# Chỉ giữ lại Students (chiếm 99.9%)
df = df[df['Profession'] == 'Student'].copy()

# Drop các cột không liên quan cho Students
drop_cols = ['id', 'Work Pressure', 'Job Satisfaction', 'Profession', 'City']
df = df.drop(columns=drop_cols, errors='ignore')
```

#### 1.2 Handle Missing Values
| Strategy | Áp dụng cho |
|----------|-------------|
| **Mode imputation** | Categorical: `Gender`, `Dietary Habits`, `Degree`, etc. |
| **Median imputation** | Numerical: `Age`, `CGPA`, `Financial Stress`, etc. |
| **Drop rows** | Nếu missing > 30% trong một row |

#### 1.3 Encode Categorical Variables
| Column | Encoding |
|--------|----------|
| `Sleep Duration` | **Ordinal**: 'Less than 5 hours' < '5-6 hours' < '7-8 hours' < 'More than 8 hours' → (0, 1, 2, 3) |
| `Dietary Habits` | **Ordinal**: Unhealthy < Moderate < Healthy → (0, 1, 2) |
| `Gender` | **Binary**: Male/Female → (0, 1) |
| `Degree` | **One-Hot Encoding** (nhiều categories) |
| `Family History of Mental Illness` | **Binary**: No/Yes → (0, 1) |
| `Have you ever had suicidal thoughts ?` | **Binary**: No/Yes → (0, 1) |

---

### Phase 2: Feature Engineering

#### 2.1 Create Composite Features
```python
# Lifestyle Score (Sleep + Diet combined)
df['Lifestyle_Score'] = df['Sleep_Encoded'] + df['Diet_Encoded']

# Total Stress Score
df['Total_Stress'] = df['Academic Pressure'] + df['Financial Stress']

# Study Efficiency (CGPA per Work/Study Hour)
df['Study_Efficiency'] = df['CGPA'] / (df['Work/Study Hours'] + 1)

# Is_High_Risk_Group (Class 12 binary flag)
df['Is_Class12'] = (df['Degree'] == 'Class 12').astype(int)
```

#### 2.2 Create Interaction Features
```python
# Academic Pressure × Lifestyle
df['AcademicPressure_x_Lifestyle'] = df['Academic Pressure'] * df['Lifestyle_Score']

# Financial Stress × Family History
df['FinancialStress_x_FamilyHistory'] = df['Financial Stress'] * df['Family_History_Encoded']
```

#### 2.3 Binning / Discretization
```python
# Age Groups
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 25, 30, 100], labels=['Teen', 'Young Adult', 'Adult', 'Mature'])

# CGPA Categories
df['CGPA_Category'] = pd.cut(df['CGPA'], bins=[0, 6, 7.5, 9, 10], labels=['Low', 'Medium', 'High', 'Excellent'])
```

---

### Phase 3: Modeling Strategy

#### 3.1 Train/Test Split
```python
from sklearn.model_selection import train_test_split, StratifiedKFold

# 80/20 split với stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5-Fold Cross Validation cho model selection
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

#### 3.2 Models to Train (By Complexity)

| Level | Model | Purpose |
|-------|-------|---------|
| **Baseline** | Logistic Regression | Interpretability, baseline performance |
| **Tree-based** | Decision Tree | Rule extraction, visualization |
| **Ensemble** | Random Forest | Feature importance, robust performance |
| **Gradient Boosting** | XGBoost / LightGBM | Best performance |
| **Interpretable** | SHAP values | Explain predictions |

#### 3.3 Evaluation Metrics

| Metric | Lý do quan trọng |
|--------|------------------|
| **Accuracy** | Overall performance |
| **Recall (Sensitivity)** | ⚠️ **Quan trọng nhất** - Không bỏ sót sinh viên có nguy cơ trầm cảm |
| **Precision** | Tránh false alarms |
| **F1-Score** | Balance Precision-Recall |
| **AUC-ROC** | Overall discrimination ability |
| **Confusion Matrix** | Detailed error analysis |

#### 3.4 Addressing Class Imbalance (nếu có)
```python
# Option 1: Class weights
model = LogisticRegression(class_weight='balanced')

# Option 2: SMOTE oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

---

## 4. 📁 Proposed Project Structure

```
project/
├── data/
│   └── student_depression_dataset.csv    # Raw data
├── notebooks/
│   ├── 01_exploration.ipynb              # ✅ EDA (done)
│   ├── 02_preprocessing.ipynb            # 🔜 Data cleaning & feature engineering
│   ├── 03_modeling.ipynb                 # 🔜 Model training & evaluation
│   └── 04_analysis.ipynb                 # 🔜 Research questions analysis
├── src/
│   ├── preprocessing.py                  # Preprocessing functions
│   ├── features.py                       # Feature engineering functions
│   └── models.py                         # Model training utilities
├── reports/
│   ├── figures/                          # Saved visualizations
│   └── final_report.pdf                  # Final report
└── INSIGHTS.md                           # This file
```

---

## 5. 📝 Next Steps Checklist

- [ ] **Phase 1: Preprocessing**
  - [ ] Load và clean data
  - [ ] Handle missing values
  - [ ] Encode categorical variables
  - [ ] Create notebook `02_preprocessing.ipynb`

- [ ] **Phase 2: Feature Engineering**
  - [ ] Create composite features (Lifestyle Score, Total Stress, etc.)
  - [ ] Create interaction features
  - [ ] Feature selection (correlation analysis, VIF for multicollinearity)

- [ ] **Phase 3: Modeling**
  - [ ] Train baseline (Logistic Regression)
  - [ ] Train tree-based models (Decision Tree, Random Forest)
  - [ ] Train gradient boosting (XGBoost/LightGBM)
  - [ ] Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
  - [ ] Model comparison & selection

- [ ] **Phase 4: Analysis & Interpretation**
  - [ ] Feature importance analysis
  - [ ] SHAP values interpretation
  - [ ] Answer Research Questions (RQ1-RQ6)
  - [ ] Risk profile clustering

- [ ] **Phase 5: Reporting**
  - [ ] Create visualizations
  - [ ] Write final report
