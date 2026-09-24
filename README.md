# AI-Powered Student Performance & Placement Analytics

**Author:** Anushka  
**Program:** IBM SkillsBuild Data Analytics with AI Academic Internship – BharatCares / AICTE  
**Dataset:**-  https://www.kaggle.com/datasets/ruchikakumbhar/placement-prediction-dataset
placementdata.csv — 10,000 students, 12 features

---

## Problem Statement

Campus placement outcomes depend on a combination of academic performance, aptitude, soft skills, and extracurricular engagement. This project builds a binary classification system to predict whether a student will be placed or not placed, using machine learning applied to a structured student dataset.

---

## Objective

- Explore and analyse the dataset to understand the factors associated with placement outcomes.
- Preprocess the data and engineer features suitable for machine learning.
- Train and evaluate six classification models.
- Identify evaluation results across multiple metrics for each model.
- Produce a clean, reproducible analysis pipeline suitable for an academic report.

---

## Dataset Description

| Property | Value |
|----------|-------|
| File | placementdata.csv |
| Source | Kaggle |
| Rows | 10,000 |
| Columns | 12 |
| Target | PlacementStatus (Placed / NotPlaced) |
| Missing values | 0 |
| Duplicate rows | 0 |

### Columns

| Column | Type | Description |
|--------|------|-------------|
| StudentID | Identifier | Dropped before modelling |
| CGPA | Numerical | College GPA (6.5 – 9.1) |
| Internships | Numerical | Count of internships (0–2) |
| Projects | Numerical | Count of projects (0–3) |
| Workshops/Certifications | Numerical | Count of workshops/certs (0–3) |
| AptitudeTestScore | Numerical | Aptitude score (60–90) |
| SoftSkillsRating | Numerical | Soft skills rating (3.0–4.8) |
| ExtracurricularActivities | Categorical | Yes / No |
| PlacementTraining | Categorical | Yes / No |
| SSC_Marks | Numerical | Class 10 percentage (55–90) |
| HSC_Marks | Numerical | Class 12 percentage (57–88) |
| PlacementStatus | Target | Placed / NotPlaced |

---

## Technologies and Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.x | Core language |
| pandas | ≥2.0 | Data manipulation |
| numpy | ≥1.24 | Numerical operations |
| matplotlib | ≥3.7 | Visualisation |
| seaborn | ≥0.13 | Statistical visualisation |
| scikit-learn | ≥1.3 | ML models, preprocessing, evaluation |
| statsmodels | ≥0.14 | VIF calculation |
| scipy | ≥1.11 | KDE for plots |
| joblib | ≥1.3 | Model serialisation |

---

## Project Workflow

```
Step 1: Dataset Understanding
  └── Shape, dtypes, missing values, duplicates, target distribution

Step 2: Exploratory Data Analysis
  └── 7 figures: distributions, correlations, feature-vs-target comparisons

Step 3: Data Preprocessing
  └── Drop StudentID, rename column, binary encode categoricals,
      encode target, stratified 80/20 split, StandardScaler

Step 4: Feature Analysis
  └── Random Forest + Gradient Boosting + Permutation importance,
      Pearson correlation, VIF — all 10 features retained

Step 5: Model Building
  └── 6 classifiers trained: LR, DT, RF, GB, SVM, KNN

Step 6: Model Evaluation
  └── Accuracy, Precision, Recall, F1, AUC, confusion matrices,
      ROC curves, 5-fold stratified CV
```

---

## EDA Key Findings

- **Target distribution:** 58.03% Not Placed, 41.97% Placed (mild class imbalance)
- **AptitudeTestScore:** Mean 84.5 (Placed) vs 75.8 (Not Placed) — strongest numerical predictor (r = 0.52)
- **HSC_Marks:** Mean 79.4 (Placed) vs 71.1 (Not Placed); second strongest (r = 0.51)
- **ExtracurricularActivities:** 62% placement rate (Yes) vs 14% (No)
- **PlacementTraining:** 52% placement rate (Yes) vs 16% (No)
- **CGPA ≥ 8.0:** Placement rate jumps from ~38% to 67–79%
- **AptitudeTestScore ≥ 85:** 77.5% placement rate vs 4.3% below 65

---

## Feature Analysis Findings

| Avg Rank | Feature | Signal |
|:---:|---|---|
| 1.7 | AptitudeTestScore | Strongest predictor across all 3 methods |
| 2.0 | HSC_Marks | Strongest tree-split feature |
| 3.0 | ExtracurricularActivities | Strongest binary categorical |
| 5.0 | SSC_Marks | Consistent mid-tier |
| 5.7 | Projects | Strong in trees |
| 6.7 | CGPA | Overall academic average |
| 7.0 | PlacementTraining | High direct impact |
| 7.0 | Workshops_Certifications | Upskilling proxy |
| 7.3 | SoftSkillsRating | Soft skills proxy |
| 9.7 | Internships | Weakest but retained |

- Highest Pearson |r| between any two features: 0.565 — well below the 0.70 concern threshold
- All VIF values: 1.19 – 1.98 — no multicollinearity
- **All 10 features retained** for model training

---

## Model Comparison Table

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Overfit Gap | CV Acc | CV F1 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 80.85% | 76.95% | 77.59% | 77.27% | 0.8837 | −0.011 | 80.00% | 76.30% |
| Decision Tree | 75.00% | 71.32% | 67.58% | 69.40% | 0.7968 | 0.128 | 75.11% | 69.85% |
| Random Forest | 79.70% | 76.56% | 74.37% | 75.45% | 0.8781 | 0.079 | 79.66% | 75.16% |
| Gradient Boosting | 79.55% | 75.84% | 75.21% | 75.52% | 0.8742 | 0.057 | 79.20% | 74.74% |
| SVM | 80.05% | 77.57% | 73.78% | 75.63% | 0.8608 | 0.008 | 79.76% | 75.37% |
| KNN | 78.95% | 75.68% | 73.42% | 74.53% | 0.8649 | 0.025 | 78.60% | 74.04% |

---

## Key Evaluation Observations

- **Logistic Regression** achieved the highest test accuracy (80.85%) and ROC-AUC (0.8837) with no train-test gap, indicating strong generalisation.
- **SVM** had the smallest overfit gap (0.008) and the second-highest test accuracy (80.05%).
- **Decision Tree** showed the highest overfit gap (0.128) and lowest test accuracy (75.00%); max_depth regularisation is recommended.
- **Random Forest and Gradient Boosting** showed moderate overfitting; hyperparameter tuning could improve test performance.
- All models outperform a baseline random classifier (AUC = 0.50) by a significant margin.

---

## Limitations

- The dataset contains only students who participated in a placement process — results may not generalise to all student populations.
- Feature ranges are compressed (e.g., CGPA starting at 6.5, Aptitude starting at 60), suggesting a pre-filtered cohort.
- Correlation between features and placement outcome does not imply causation.
- No hyperparameter tuning was performed; further optimisation may improve model performance.
- The dataset does not include contextual factors such as company type, branch/stream, or economic background.

---

## Conclusion

This project demonstrates that student placement outcomes can be predicted with reasonable accuracy using a combination of academic, aptitude, and engagement features. Logistic Regression and SVM show the best combination of accuracy, AUC, and generalisation on this dataset. The analysis pipeline — from EDA through preprocessing, feature analysis, model training, and evaluation — provides a reproducible foundation for further research.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open the notebook
jupyter notebook Anushka_StudentPerformanceAnalytics.ipynb

# 3. Run all cells in order (Kernel > Restart & Run All)
#    The dataset file placementdata.csv must be in the same folder.
```

---

---

## Web Application — Student Placement Prediction & AI Career Insights

### Overview

A Streamlit web application that:
- Takes student profile inputs via a clean form
- Runs inference using the saved Logistic Regression model and StandardScaler
- Displays prediction, probability, and a student profile summary
- Optionally calls **Google Gemini** to explain the result and generate personalised career recommendations

### Quick Start

```bash
# 1. Install dependencies (if not already done)
pip install -r requirements.txt

# 2. (Optional) Set Gemini API key for AI insights
#    Windows PowerShell:
$env:GEMINI_API_KEY = "your_gemini_api_key_here"
#    macOS / Linux:
export GEMINI_API_KEY="your_gemini_api_key_here"

# 3. Run the app
streamlit run app.py
```

The app will open in your browser at **http://localhost:8501**

> The models/ folder must be present with `logistic_regression.pkl` and `scaler.pkl`.  
> The dataset file must be in the same folder as app.py.

### GenAI Setup

- Get a free API key at: https://aistudio.google.com/app/apikey
- Set it as the environment variable `GEMINI_API_KEY` before running the app
- The app works fully without a key — AI insights section will show an informational message instead

### Application Features

| Feature | Description |
|---------|-------------|
| Input form | 10 student features with dropdowns and number inputs |
| ML Prediction | Placed / Not Placed with probability bar |
| Profile Summary | 9 metric cards showing all entered values |
| AI Explanation | Gemini explains why the model predicted this result |
| AI Recommendations | 4 personalised, actionable improvement tips |
| Graceful fallback | App works without GEMINI_API_KEY |

### Files Added for Web App

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web application (single file) |
| `requirements.txt` | Updated with `streamlit` and `google-genai` |

