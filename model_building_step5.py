# =============================================================================
# STEP 5: Machine Learning Model Building
# Project  : AI-Powered Student Performance & Placement Analytics
# Author   : Anushka
# Program  : IBM SkillsBuild Data Analytics with AI - BharatCares / AICTE
# Dataset  : placementdata.csv  (10,000 students, 12 features)
# =============================================================================

# -- Imports ------------------------------------------------------------------
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =============================================================================
# REPRODUCE PREPROCESSING (mirrors preprocessing_step3.py exactly)
# -- In the notebook this block can be replaced by %run preprocessing_step3.py
# =============================================================================
df = pd.read_csv("placementdata.csv")
df = df.drop(columns=["StudentID"])
df = df.rename(columns={"Workshops/Certifications": "Workshops_Certifications"})
df["ExtracurricularActivities"] = df["ExtracurricularActivities"].map({"Yes": 1, "No": 0})
df["PlacementTraining"]         = df["PlacementTraining"].map({"Yes": 1, "No": 0})
df["PlacementStatus"]           = df["PlacementStatus"].map({"Placed": 1, "NotPlaced": 0})

FEATURE_COLS = [
    "CGPA", "Internships", "Projects", "Workshops_Certifications",
    "AptitudeTestScore", "SoftSkillsRating",
    "ExtracurricularActivities", "PlacementTraining",
    "SSC_Marks", "HSC_Marks",
]
TARGET_COL = "PlacementStatus"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index
)

print("Preprocessing complete.")
print(f"  Train : {X_train.shape[0]:,} rows   Test : {X_test.shape[0]:,} rows")
print(f"  Features : {len(FEATURE_COLS)}")
print()

# =============================================================================
# MODEL REGISTRY
# Each entry: (name, estimator, uses_scaled_data)
# =============================================================================
MODELS = [
    (
        "Logistic Regression",
        LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
            C=1.0
        ),
        True,    # requires scaled data
    ),
    (
        "Decision Tree",
        DecisionTreeClassifier(
            max_depth=None,        # fully grown; pruning explored in Step 6
            min_samples_leaf=5,
            random_state=42
        ),
        False,
    ),
    (
        "Random Forest",
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        False,
    ),
    (
        "Gradient Boosting",
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ),
        False,
    ),
    (
        "SVM",
        CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
            ensemble=False
        ),
        True,    # requires scaled data
    ),
    (
        "KNN",
        KNeighborsClassifier(
            n_neighbors=11,        # odd number, avoids ties; tuned in Step 6
            metric="minkowski",
            p=2
        ),
        True,    # requires scaled data
    ),
]

# =============================================================================
# CELL 5.1 -- Model Configuration Summary
# =============================================================================
print("=" * 70)
print("  5.1  MODEL CONFIGURATION SUMMARY")
print("=" * 70)
print()
print(f"  {'Model':<25} {'Data':<10} {'Key Hyperparameters'}")
print("  " + "-" * 68)
configs = {
    "Logistic Regression": "max_iter=1000, C=1.0, solver='lbfgs'",
    "Decision Tree":       "min_samples_leaf=5, max_depth=None",
    "Random Forest":       "n_estimators=300, min_samples_leaf=5",
    "Gradient Boosting":   "n_estimators=200, lr=0.10, max_depth=4",
    "SVM":                 "kernel='rbf', C=1.0, gamma='scale'",
    "KNN":                 "n_neighbors=11, metric='minkowski' (p=2)",
}
for name, scaled, *_ in [(n, s) for n, _, s in MODELS]:
    data_label = "Scaled" if scaled else "Unscaled"
    print(f"  {name:<25} {data_label:<10} {configs[name]}")
print()

# =============================================================================
# CELL 5.2 -- Train All Models
# =============================================================================
print("=" * 70)
print("  5.2  TRAINING MODELS")
print("=" * 70)
print()

results = []

for name, model, use_scaled in MODELS:
    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled  if use_scaled else X_test

    print(f"  Training : {name} ...", end=" ", flush=True)
    t0 = time.time()
    model.fit(X_tr, y_train)
    train_time = time.time() - t0

    # Predictions
    y_pred_train = model.predict(X_tr)
    y_pred_test  = model.predict(X_te)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test,  y_pred_test)

    print(f"done  ({train_time:.2f}s)  "
          f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    results.append({
        "Model":           name,
        "Data":            "Scaled" if use_scaled else "Unscaled",
        "Train_Samples":   len(y_train),
        "Test_Samples":    len(y_test),
        "Train_Acc":       round(train_acc, 4),
        "Test_Acc":        round(test_acc,  4),
        "Train_Time_s":    round(train_time, 3),
        "estimator":       model,          # kept for Step 6
    })

print()

# =============================================================================
# CELL 5.3 -- Training Summary Table
# =============================================================================
print("=" * 70)
print("  5.3  MODEL TRAINING SUMMARY")
print("=" * 70)
print()

res_df = pd.DataFrame(results).drop(columns=["estimator"])

header = (f"  {'Model':<25} {'Data':<10} {'Train Acc':>10} "
          f"{'Test Acc':>10} {'Time (s)':>10}")
print(header)
print("  " + "-" * 68)
for _, row in res_df.iterrows():
    print(f"  {row['Model']:<25} {row['Data']:<10} "
          f"{row['Train_Acc']:>10.4f} {row['Test_Acc']:>10.4f} "
          f"{row['Train_Time_s']:>10.3f}")

print()
print("  NOTE: Detailed evaluation (confusion matrix, precision, recall,")
print("        F1-score, ROC-AUC, cross-validation) is performed in Step 6.")
print("        No model is declared best at this stage.")
print()

# =============================================================================
# CELL 5.4 -- Overfitting Check (train vs test accuracy gap)
# =============================================================================
print("=" * 70)
print("  5.4  OVERFITTING INDICATOR (Train - Test Accuracy Gap)")
print("=" * 70)
print()
print(f"  {'Model':<25} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>8}  {'Signal'}")
print("  " + "-" * 66)
for _, row in res_df.iterrows():
    gap = row["Train_Acc"] - row["Test_Acc"]
    if gap > 0.10:
        signal = "HIGH overfit -- regularise in Step 6"
    elif gap > 0.05:
        signal = "Moderate -- monitor"
    else:
        signal = "OK -- generalises well"
    print(f"  {row['Model']:<25} {row['Train_Acc']:>10.4f} "
          f"{row['Test_Acc']:>10.4f} {gap:>8.4f}  {signal}")

print()

# =============================================================================
# CELL 5.5 -- Persist Trained Models and Splits for Step 6
# =============================================================================
import joblib, os

os.makedirs("models", exist_ok=True)

for entry in results:
    safe_name = entry["Model"].replace(" ", "_").lower()
    joblib.dump(entry["estimator"], f"models/{safe_name}.pkl")

joblib.dump(scaler,        "models/scaler.pkl")
joblib.dump(X_train,       "models/X_train.pkl")
joblib.dump(X_test,        "models/X_test.pkl")
joblib.dump(X_train_scaled,"models/X_train_scaled.pkl")
joblib.dump(X_test_scaled, "models/X_test_scaled.pkl")
joblib.dump(y_train,       "models/y_train.pkl")
joblib.dump(y_test,        "models/y_test.pkl")

saved = os.listdir("models")
print("=" * 70)
print("  5.5  SAVED ARTEFACTS (models/ folder)")
print("=" * 70)
print()
for f in sorted(saved):
    size_kb = round(os.path.getsize(f"models/{f}") / 1024, 1)
    print(f"    models/{f:<45} {size_kb:>8.1f} KB")

print()
print("  All models and data splits saved for Step 6.")
print()

# =============================================================================
# CELL 5.6 -- Step Summary
# =============================================================================
print("=" * 70)
print("  5.6  STEP 5 COMPLETE -- SUMMARY")
print("=" * 70)
print("""
  Models trained  : 6
  Training set    : 8,000 samples
  Test set        : 2,000 samples
  Features used   : 10 (all retained from Step 4)

  Scaling applied:
    Scaled   (mean=0, std=1) -> Logistic Regression, SVM, KNN
    Unscaled                 -> Decision Tree, Random Forest, Gradient Boosting

  What happens next (Step 6 -- Model Evaluation & Comparison):
    - Confusion matrix for each model
    - Precision, Recall, F1-score (class-level and macro/weighted)
    - ROC-AUC score and ROC curves
    - 5-fold stratified cross-validation
    - Best model selection and justification
""")
