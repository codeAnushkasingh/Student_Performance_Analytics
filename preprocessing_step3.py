# =============================================================================
# STEP 3: Data Preprocessing
# Project  : AI-Powered Student Performance & Placement Analytics
# Author   : Anushka
# Program  : IBM SkillsBuild Data Analytics with AI - BharatCares / AICTE
# Dataset  : placementdata.csv  (10,000 students, 12 features)
# =============================================================================

# -- Imports ------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# 3.1  Load Raw Data
# -----------------------------------------------------------------------------
df_raw = pd.read_csv("placementdata.csv")

print("=" * 60)
print("  3.1  RAW DATASET")
print("=" * 60)
print(f"  Shape   : {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
print(f"  Columns : {list(df_raw.columns)}")
print()

# -----------------------------------------------------------------------------
# 3.2  Final Data-Quality Check
#      (missing values, duplicates, invalid ranges)
# -----------------------------------------------------------------------------
df = df_raw.copy()

print("=" * 60)
print("  3.2  FINAL DATA-QUALITY CHECK")
print("=" * 60)

# Missing values
missing = df.isnull().sum()
print("\n  Missing values per column:")
print("  " + missing.to_string().replace("\n", "\n  "))
print(f"\n  Total missing cells : {missing.sum()}")

# Duplicates
n_dupes = df.duplicated().sum()
print(f"  Duplicate rows      : {n_dupes}")

# Invalid-range check for key numerical columns
range_checks = {
    "CGPA":                      (0, 10),
    "AptitudeTestScore":         (0, 100),
    "SoftSkillsRating":          (0, 5),
    "SSC_Marks":                 (0, 100),
    "HSC_Marks":                 (0, 100),
    "Internships":               (0, 10),
    "Projects":                  (0, 20),
    "Workshops/Certifications":  (0, 20),
}

print("\n  Range / impossible-value check:")
all_clean = True
for col, (lo, hi) in range_checks.items():
    bad = df[(df[col] < lo) | (df[col] > hi)].shape[0]
    status = "[OK]" if bad == 0 else f"[FAIL] {bad} invalid rows"
    print(f"    {col:<30} [{lo}, {hi}]  {status}")
    if bad > 0:
        all_clean = False

# Categorical consistency
for col in ["ExtracurricularActivities", "PlacementTraining", "PlacementStatus"]:
    vals = df[col].unique().tolist()
    print(f"    {col:<30} values -> {vals}")

print()
if missing.sum() == 0 and n_dupes == 0 and all_clean:
    print("  [PASS] No missing values, no duplicates, no invalid entries.")
    print("         No imputation or row removal required.\n")
else:
    # Safety net -- should not trigger on this dataset
    df = df.dropna()
    df = df.drop_duplicates()
    print("  [WARN] Issues found -- rows with problems dropped.\n")

# -----------------------------------------------------------------------------
# 3.3  Remove Identifier Column
# -----------------------------------------------------------------------------
print("=" * 60)
print("  3.3  REMOVE IDENTIFIER COLUMN")
print("=" * 60)

df = df.drop(columns=["StudentID"])
print("  Dropped : 'StudentID'  (sequential ID, no predictive value)")
print(f"  Remaining columns : {df.shape[1]}\n")

# -----------------------------------------------------------------------------
# 3.4  Rename Column with Special Character
# -----------------------------------------------------------------------------
print("=" * 60)
print("  3.4  RENAME COLUMN (fix '/' in name)")
print("=" * 60)

df = df.rename(columns={"Workshops/Certifications": "Workshops_Certifications"})
print("  'Workshops/Certifications' -> 'Workshops_Certifications'")
print("  Reason: '/' causes parsing errors in some ML libraries.\n")

# -----------------------------------------------------------------------------
# 3.5  Encode Categorical Features (Yes/No -> 1/0)
# -----------------------------------------------------------------------------
print("=" * 60)
print("  3.5  ENCODE CATEGORICAL FEATURES")
print("=" * 60)

binary_map = {"Yes": 1, "No": 0}

df["ExtracurricularActivities"] = df["ExtracurricularActivities"].map(binary_map)
df["PlacementTraining"]         = df["PlacementTraining"].map(binary_map)

print("  Binary encoding applied (Yes=1, No=0):")
print("    ExtracurricularActivities : Yes -> 1  |  No -> 0")
print("    PlacementTraining         : Yes -> 1  |  No -> 0")
print()
print("  Encoded value counts:")
for col in ["ExtracurricularActivities", "PlacementTraining"]:
    vc = df[col].value_counts().to_dict()
    print(f"    {col}: {vc}")
print()

# -----------------------------------------------------------------------------
# 3.6  Encode Target Variable (PlacementStatus)
# -----------------------------------------------------------------------------
print("=" * 60)
print("  3.6  ENCODE TARGET VARIABLE")
print("=" * 60)

target_map = {"Placed": 1, "NotPlaced": 0}
df["PlacementStatus"] = df["PlacementStatus"].map(target_map)

print("  PlacementStatus encoding:  Placed=1  |  NotPlaced=0")
vc_target = df["PlacementStatus"].value_counts().to_dict()
print(f"  Encoded counts : {vc_target}")
pct_placed    = df["PlacementStatus"].mean() * 100
pct_notplaced = 100 - pct_placed
print(f"  Class balance  : Placed={pct_placed:.2f}%  |  NotPlaced={pct_notplaced:.2f}%\n")

# -----------------------------------------------------------------------------
# 3.7  Inspect Processed DataFrame
# -----------------------------------------------------------------------------
print("=" * 60)
print("  3.7  PROCESSED DATAFRAME OVERVIEW")
print("=" * 60)

print(f"\n  Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\n  Data types after preprocessing:")
print("  " + df.dtypes.to_string().replace("\n", "\n  "))
print("\n  First 3 rows:")
print(df.head(3).to_string(index=False))
print()

# -----------------------------------------------------------------------------
# 3.8  Prepare Features (X) and Target (y)
# -----------------------------------------------------------------------------
print("=" * 60)
print("  3.8  PREPARE X (FEATURES) AND y (TARGET)")
print("=" * 60)

FEATURE_COLS = [
    "CGPA",
    "Internships",
    "Projects",
    "Workshops_Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "ExtracurricularActivities",
    "PlacementTraining",
    "SSC_Marks",
    "HSC_Marks",
]
TARGET_COL = "PlacementStatus"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print(f"\n  Features (X) : {X.shape[1]} columns x {X.shape[0]:,} rows")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"    {i:2d}. {col}")

print(f"\n  Target (y)   : '{TARGET_COL}'  (0 = NotPlaced, 1 = Placed)")
print(f"  Shape        : ({y.shape[0]:,},)\n")

# -----------------------------------------------------------------------------
# 3.9  Train / Test Split  (Stratified, 80% train, 20% test)
# -----------------------------------------------------------------------------
print("=" * 60)
print("  3.9  STRATIFIED TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # preserves class ratio in both splits
)

def class_balance(y_series, label):
    vc  = y_series.value_counts().to_dict()
    n   = len(y_series)
    p   = vc.get(1, 0) / n * 100
    np_ = vc.get(0, 0) / n * 100
    print(f"  {label:<22} total={n:,}  "
          f"Placed={vc.get(1,0):,} ({p:.2f}%)  "
          f"NotPlaced={vc.get(0,0):,} ({np_:.2f}%)")

print()
class_balance(y,       "Full dataset")
class_balance(y_train, "Training set")
class_balance(y_test,  "Test set    ")

print()
print("  [OK] Stratified split preserves the original ~58% / ~42% class ratio")
print("       in both the training and test sets.\n")

# -----------------------------------------------------------------------------
# 3.10  Feature Scaling (StandardScaler)
#       Scaled copy  -> for distance-based models (LR, SVM, KNN)
#       Unscaled     -> for tree-based models (Decision Tree, Random Forest)
# -----------------------------------------------------------------------------
print("=" * 60)
print("  3.10  FEATURE SCALING")
print("=" * 60)
print("""
  WHY SCALING IS NEEDED FOR SOME MODELS
  --------------------------------------
  Models sensitive to feature magnitude (Logistic Regression, SVM, KNN)
  require all features on a comparable scale. Without scaling, a feature
  like AptitudeTestScore (60-90) dominates SoftSkillsRating (3.0-4.8).

  STRATEGY
  ---------
  - StandardScaler (z-score): transforms each feature to mean=0, std=1.
  - Fit ONLY on X_train, then transform both X_train and X_test.
    (Fitting on test data would leak information -- data leakage.)

  MODEL SCALING REQUIREMENTS
  ---------------------------
  Model                      Scaling Required?
  -----------------------------------------------
  Logistic Regression        Yes
  K-Nearest Neighbours (KNN) Yes
  Support Vector Machine     Yes
  Decision Tree              No  (splits on thresholds)
  Random Forest              No  (ensemble of trees)
  Gradient Boosting / XGBoost No  (ensemble of trees)
  Naive Bayes                No  (probabilistic, not distance-based)
  -----------------------------------------------
  Both scaled and unscaled sets are prepared for Step 4.
""")

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=FEATURE_COLS,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=FEATURE_COLS,
    index=X_test.index
)

print("  Scaler fitted on X_train only (no data leakage).")
print("  X_train_scaled and X_test_scaled are ready.\n")

print("  Scaled feature statistics on training set (should be ~mean=0, std=1):")
summary = X_train_scaled.describe().loc[["mean", "std"]].round(4)
print("  " + summary.to_string().replace("\n", "\n  "))
print()

# -----------------------------------------------------------------------------
# 3.11  Final Summary
# -----------------------------------------------------------------------------
print("=" * 60)
print("  3.11  PREPROCESSING SUMMARY")
print("=" * 60)
print("""
  Step   Action                              Result
  ------------------------------------------------------------
  3.1    Load raw data                       10,000 rows x 12 cols
  3.2    Quality check                       0 missing, 0 dupes, 0 invalid
  3.3    Drop StudentID                      11 cols remaining
  3.4    Rename Workshops/Certifications     No special chars in names
  3.5    Encode Yes/No features              2 binary cols encoded (0/1)
  3.6    Encode target PlacementStatus       Placed=1 | NotPlaced=0
  3.7    Final processed shape               10,000 rows x 11 cols
  3.8    Prepare X, y                        X: 10 features | y: 1 target
  3.9    Stratified split (80/20)            Train: 8,000 | Test: 2,000
  3.10   StandardScaler (train-fit only)     X_train_scaled | X_test_scaled
  ------------------------------------------------------------

  Objects ready for Step 4 (Model Training):
    X_train        -- unscaled  (use for Decision Tree, Random Forest)
    X_test         -- unscaled
    X_train_scaled -- scaled    (use for Logistic Regression, SVM, KNN)
    X_test_scaled  -- scaled
    y_train        -- target labels for training
    y_test         -- target labels for evaluation
    scaler         -- fitted StandardScaler instance (save for deployment)
""")
