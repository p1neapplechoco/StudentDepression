# Student Depression Analysis

## Project Overview

This project analyzes factors contributing to depression in students using machine learning and statistical methods. The analysis investigates 10 research questions across three themes: Lifestyle Factors, Psychological Pressure, and Academic & Work Variables.

**Team Members:**
- P4DS Course Project Team
- University Data Science Program

## Dataset

### Source
- **Platform:** Kaggle
- **Link:** [Student Depression Dataset](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset/data)
- **Author:** Adil Shamim
- **License:** Apache License 2.0
- **Usability Score:** 10.0/10

### Dataset Description
- **Size:** 27,901 records × 18 features
- **Geographic Scope:** 52 cities across India
- **Data Type:** Self-reported survey data
- **Data Quality:** No missing values, no duplicate records

### Target Variable
- **Depression** (Binary classification)
  - Class 0 (No Depression): 11,565 students (41.5%)
  - Class 1 (Depression): 16,336 students (58.5%)

### Features
- **Demographics:** Age, Gender, City, Profession
- **Academic:** CGPA, Degree, Academic Pressure, Study Satisfaction
- **Work:** Work Pressure, Job Satisfaction, Work/Study Hours
- **Lifestyle:** Sleep Duration, Dietary Habits
- **Psychological:** Suicidal Thoughts, Financial Stress, Family History of Mental Illness

## Research Questions

### Theme 1: Lifestyle Factors

**RQ1: The 7-8 Hour Sleep Paradox**
- Why do students sleeping the medically recommended 7-8 hours have HIGHER depression rates (60.67%) than those sleeping 5-6 hours (58.72%)?

**RQ2: Diet as Sleep Compensation**
- Can a healthy diet compensate for sleep deprivation? Does the improvement differ across sleep duration groups?

**RQ3: Optimal Lifestyle Threshold**
- What is the "sweet spot" in the Sleep × Diet matrix that minimizes depression risk?

### Theme 2: Psychological Pressure Factors

**RQ4: Cumulative Pressure Effect**
- Does the combination of Academic Pressure, Work Pressure, and Financial Stress increase suicidal thoughts exponentially?

**RQ5: Satisfaction as Protective Factor**
- Can high Study Satisfaction reduce suicidal thoughts in students with Family History of Mental Illness to levels comparable to those without?

**RQ6: Academic Pressure Danger Threshold**
- At what Academic Pressure level does the probability of suicidal thoughts exceed 50%?

### Theme 3: Academic & Work Variables

**RQ7: Academic Factors Impact**
- Between CGPA and Work/Study Hours, which factor has stronger impact on depression?

**RQ8: High-Risk Degree Programs**
- Which degree programs have depression rates significantly higher than average (58.5%)?

**RQ9: Work/Study Hours Safe Threshold**
- At what daily work/study hour threshold does depression probability exceed 50%?

**RQ10: Achievement-Satisfaction Paradox**
- Do students with high CGPA but low Study Satisfaction have higher depression rates than low CGPA students with high satisfaction?

## Key Findings

### Lifestyle Factors (RQ1-RQ3)

**RQ1: Sleep Paradox Explained**
- Finding: Reverse causation - depressed students sleep more (hypersomnia symptom)
- Chi-square test: χ² = 9.07, p = 0.0026
- Implication: Sleep quality matters more than quantity

**RQ2: Diet Compensation Effect**
- Finding: Healthy diet reduces depression by 25-33% but cannot substitute sleep
- Even with healthy diet, <5h sleep (50.38%) > >8h + Healthy (36.64%)
- Conclusion: Dual intervention (sleep + diet) is necessary

**RQ3: Optimal Sweet Spot**
- Best combination: >8 hours sleep + Healthy diet (36.64% depression)
- Worst combination: <5h + Unhealthy (75.96% depression)
- ROI: 51.8% reduction from worst to best

### Psychological Pressure (RQ4-RQ6)

**RQ4: Academic Pressure Dominates**
- Academic Pressure: OR = 1.46, p < 0.001 (46% increase per unit)
- Financial Stress: OR = 1.31, p < 0.001 (31% increase per unit)
- Work Pressure: Not significant (most students don't work)
- Effects are additive, not exponential

**RQ5: Satisfaction Eliminates Risk Gap**
- Low satisfaction: 13 pp gap between family history groups
- High satisfaction (level 5): Only 2 pp gap (not significant)
- Conclusion: Environmental interventions can overcome hereditary risk

**RQ6: Danger Threshold = 1.57**
- Suicidal thought probability exceeds 50% at Academic Pressure = 1.57
- Recommendation: Flag students at pressure ≥ 2 for intervention

### Academic Variables (RQ7-RQ10)

**RQ7: Hours Matter, CGPA Doesn't**
- Work/Study Hours: r = 0.21, OR = 1.38, p < 0.001
- CGPA: r = 0.02, p > 0.05 (not significant)
- Conclusion: Time spent studying, not grades, predicts depression

**RQ8: Class 12 Students at Highest Risk**
- Class 12: 70.8% depression (21% above average)
- Reason: High-stakes exams, peer pressure, parental expectations
- Implication: Target mental health programs for pre-university students

**RQ9: Safe Threshold < 4 Hours/Day**
- Depression probability exceeds 50% at 4 hours/day work/study
- At 12 hours/day: 73% depression probability
- Recommendation: Safe zone is < 4 hours/day

**RQ10: Satisfaction Trumps Achievement**
- High CGPA + Low Satisfaction: 67.4% depression
- Low CGPA + High Satisfaction: 52.9% depression
- Difference: 14.5 percentage points in favor of satisfaction
- Conclusion: Happiness matters more than grades

### Overall Model Results

**Best Model:** Gradient Boosting (AUC-ROC = 0.92, Accuracy = 0.89)

**Top Predictive Features:**
1. Suicidal Thoughts (importance: 0.352)
2. Academic Pressure (importance: 0.198)
3. Financial Stress (importance: 0.156)
4. Work/Study Hours (importance: 0.089)
5. Sleep Duration (importance: 0.067)

**Notable Non-Predictors:**
- CGPA (importance: 0.003) - Validates RQ7
- Family History (importance: 0.005) - Validates RQ5

## File Structure

```
StudentDepression/
├── data/
│   └── student_depression_dataset.csv       # Raw dataset (27,901 records)
├── notebooks/
│   ├── 01_exploration.ipynb                 # Initial EDA and overview
│   ├── 02_lifestyle.ipynb                   # RQ1-RQ3: Sleep & Diet analysis
│   ├── 03_pressure.ipynb                    # RQ4-RQ6: Pressure & mental health
│   ├── 04_work.ipynb                        # RQ7-RQ10: Academic & work factors
│   ├── modeling.ipynb                       # Predictive modeling
│   └── preprocessing.ipynb                  # Data cleaning demonstrations
├── src/
│   ├── preprocessing.py                     # Data cleaning functions
│   ├── features.py                          # Feature engineering pipeline
│   ├── models.py                            # ML modeling utilities
│   └── run_pipeline.py                      # Main execution script
├── results/
│   ├── model_comparison.csv                 # Model performance metrics
│   ├── feature_importance.csv               # Feature importance rankings
│   └── processed_data.csv                   # Cleaned dataset
└── README.md
```

### Notebooks Description

**01_exploration.ipynb**
- Dataset overview and quality checks
- Feature distributions and statistics
- Correlation analysis
- Initial insights

**02_lifestyle.ipynb**
- RQ1: Sleep paradox analysis
- RQ2: Diet compensation effect
- RQ3: Optimal lifestyle combination
- Statistical tests: Chi-square, Kruskal-Wallis, Mann-Whitney U

**03_pressure.ipynb**
- RQ4: Cumulative pressure effect analysis
- RQ5: Satisfaction as protective factor
- RQ6: Danger threshold identification
- Logistic regression with odds ratios

**04_work.ipynb**
- RQ7: CGPA vs Work/Study Hours impact
- RQ8: Degree-specific depression rates
- RQ9: Work/study hours threshold
- RQ10: Achievement-satisfaction paradox

**modeling.ipynb**
- Model comparison (Logistic Regression, Random Forest, Gradient Boosting, Decision Tree)
- Feature importance analysis
- Performance evaluation
- Cross-validation

**preprocessing.ipynb**
- Step-by-step data cleaning
- Encoding strategies demonstration
- Feature creation examples

### Source Code Modules

**preprocessing.py (252 lines)**
- Load and clean data
- Encode categorical variables (ordinal and binary)
- Handle sleep duration and dietary habits
- Entry point: `preprocess_pipeline(data_path)`

**features.py (181 lines)**
- Create age and CGPA groupings
- Generate satisfaction and pressure categories
- Build interaction features
- Create achievement-satisfaction composite
- Entry point: `feature_engineering_pipeline(df)`

**models.py (251 lines)**
- Train 4 classification models
- Evaluate with multiple metrics (Accuracy, AUC, F1, Precision, Recall)
- Extract feature importance
- Support cross-validation and SMOTE
- Entry point: `modeling_pipeline(X, y, ...)`

**run_pipeline.py (47 lines)**
- Orchestrate complete pipeline
- Configurable parameters (SMOTE, cross-validation, test size)
- Save results to CSV files

## How to Run

### Prerequisites

Install required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels scipy
```

Or using requirements file:

```bash
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
cd src
python run_pipeline.py
```

This will:
1. Load and preprocess data
2. Engineer features
3. Train and evaluate 4 models
4. Save results to `results/` directory

### Run Specific RQ Analysis

Open Jupyter notebooks for detailed analysis:

```bash
jupyter notebook
```

Navigate to:
- `notebooks/02_lifestyle.ipynb` for RQ1-RQ3
- `notebooks/03_pressure.ipynb` for RQ4-RQ6
- `notebooks/04_work.ipynb` for RQ7-RQ10

### Customize Pipeline Configuration

Edit `src/run_pipeline.py`:

```python
USE_SMOTE = False           # Handle class imbalance
CROSS_VALIDATE = True       # Enable 5-fold cross-validation
TEST_SIZE = 0.2             # 80-20 train-test split
```

### Use Modules Programmatically

```python
from preprocessing import preprocess_pipeline
from features import feature_engineering_pipeline, prepare_modeling_data
from models import modeling_pipeline

# Load and preprocess
df_processed, df_original = preprocess_pipeline('data/student_depression_dataset.csv')

# Engineer features
df_features = feature_engineering_pipeline(df_processed)

# Prepare for modeling
X, y, feature_names = prepare_modeling_data(df_features)

# Train models
results = modeling_pipeline(X, y, test_size=0.2)

# Access results
best_model = results['best_model']
performance_df = results['results']
feature_importance = results['feature_importance']
```

## Dependencies

### Python Version
- Python 3.7 or higher

### Core Libraries

```
pandas>=1.3.0          # Data manipulation
numpy>=1.21.0          # Numerical computing
scikit-learn>=1.0.0    # Machine learning models
matplotlib>=3.4.0      # Plotting
seaborn>=0.11.0        # Statistical visualization
statsmodels>=0.13.0    # Statistical tests for RQ analysis
scipy>=1.7.0           # Hypothesis testing
```

### Optional Libraries

```
jupyter>=1.0.0         # For running notebooks
notebook>=6.4.0        # Jupyter notebook interface
```

## Methodology

### Data Preprocessing
1. Remove single quotes from categorical values
2. Drop ID column
3. Encode categorical variables:
   - Ordinal: Sleep Duration (1-4), Dietary Habits (1-3)
   - Binary: Suicidal Thoughts, Family History, Gender (0/1)

### Feature Engineering
1. Create groupings: Age, CGPA, Satisfaction, Pressure, Hours
2. Build interaction term: Academic Pressure × Study Satisfaction
3. Generate composite: Achievement-Satisfaction categories
4. Flag high achievers: CGPA ≥ 8.0

### Statistical Analysis
- **RQ1-3:** Chi-square, Kruskal-Wallis, Mann-Whitney U tests
- **RQ4-6:** Logistic regression with odds ratios
- **RQ7-10:** Correlation, ANOVA, stratified analysis

### Machine Learning
- **Models:** Logistic Regression, Random Forest, Gradient Boosting, Decision Tree
- **Split:** 80-20 train-test with stratification
- **Scaling:** StandardScaler for Logistic Regression only
- **Evaluation:** Accuracy, AUC-ROC, F1-Score, Precision, Recall
- **Selection:** Best model by AUC-ROC score

## Results

### Model Performance

| Model               | Accuracy | AUC-ROC | F1-Score | Precision | Recall |
|---------------------|----------|---------|----------|-----------|--------|
| Gradient Boosting   | 0.8856   | 0.9245  | 0.9012   | 0.8934    | 0.9092 |
| Random Forest       | 0.8798   | 0.9198  | 0.8945   | 0.8876    | 0.9016 |
| Logistic Regression | 0.8512   | 0.8856  | 0.8634   | 0.8501    | 0.8771 |
| Decision Tree       | 0.8234   | 0.8421  | 0.8312   | 0.8198    | 0.8429 |

### Output Files

All results saved to `results/` directory:
- `model_comparison.csv` - Performance metrics for all models
- `feature_importance.csv` - Feature importance rankings
- `processed_data.csv` - Full engineered dataset (27,901 × 22 columns)

## Limitations

### Data Limitations
- Self-reported data subject to response bias
- Cross-sectional design cannot establish causality
- Geographic scope limited to India
- Depression status not clinically validated
- Single time point, no longitudinal tracking

### Methodological Limitations
- Class imbalance (58.5% vs 41.5%) may affect model performance
- Feature engineering cutoffs are somewhat arbitrary
- No external validation on independent dataset
- Multiple testing (10 RQs) may increase false discovery risk

### RQ-Specific Limitations
- **RQ1:** Cannot distinguish sleep quality from quantity
- **RQ2-3:** Self-reported diet quality, no nutritional biomarkers
- **RQ4-6:** Work Pressure has insufficient variation, suicidal thoughts are lifetime history
- **RQ7-10:** CGPA scale not universal, degree sample sizes vary

## License

- **Dataset:** Apache License 2.0 (Kaggle)
- **Code:** Educational use - P4DS University Project

## Acknowledgments

- **Dataset Author:** Adil Shamim (Kaggle)
- **Data Source:** 27,901 students across 52 cities in India
- **Libraries:** scikit-learn, statsmodels, scipy, pandas, numpy
- **Course:** P4DS - University Data Science Program

## Disclaimer

This project is for educational and research purposes only. The models and findings should not be used as a substitute for professional mental health diagnosis or treatment. If you or someone you know is experiencing depression or suicidal thoughts, please seek help from qualified mental health professionals immediately.

**Crisis Resources:**
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
- National Suicide Prevention Lifeline (US): 988
- Samaritans (UK): 116 123

## Contact

For questions about the project, please refer to the course materials or contact the project team through the university.
