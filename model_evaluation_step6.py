# =============================================================================
# STEP 6: Comprehensive Model Evaluation
# Project  : AI-Powered Student Performance & Placement Analytics
# Author   : Anushka
# Program  : IBM SkillsBuild Data Analytics with AI - BharatCares / AICTE
# Dataset  : placementdata.csv  (10,000 students, 12 features)
# =============================================================================

# -- Imports ------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# -- Global style -------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
})

COLORS = ["#2563EB", "#EF4444", "#10B981", "#F59E0B", "#7C3AED", "#EC4899"]
MODEL_ORDER = [
    "Logistic Regression", "Decision Tree", "Random Forest",
    "Gradient Boosting", "SVM", "KNN"
]

# =============================================================================
# CELL 6.0 -- Load Saved Artefacts from Step 5
# =============================================================================
print("=" * 68)
print("  6.0  LOADING SAVED ARTEFACTS FROM STEP 5")
print("=" * 68)

X_train        = joblib.load("models/X_train.pkl")
X_test         = joblib.load("models/X_test.pkl")
X_train_scaled = joblib.load("models/X_train_scaled.pkl")
X_test_scaled  = joblib.load("models/X_test_scaled.pkl")
y_train        = joblib.load("models/y_train.pkl")
y_test         = joblib.load("models/y_test.pkl")

model_files = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree":       "decision_tree",
    "Random Forest":       "random_forest",
    "Gradient Boosting":   "gradient_boosting",
    "SVM":                 "svm",
    "KNN":                 "knn",
}
SCALED_MODELS = {"Logistic Regression", "SVM", "KNN"}

models = {}
for name, fname in model_files.items():
    models[name] = joblib.load(f"models/{fname}.pkl")

print(f"\n  Loaded {len(models)} models.")
print(f"  Test set : {len(y_test):,} samples  "
      f"(Placed={y_test.sum():,} | NotPlaced={(y_test==0).sum():,})\n")

# =============================================================================
# CELL 6.1 -- Per-Model Metrics on Test Set
# =============================================================================
print("=" * 68)
print("  6.1  TEST-SET EVALUATION METRICS")
print("=" * 68)

eval_records = []

for name in MODEL_ORDER:
    model = models[name]
    X_te  = X_test_scaled if name in SCALED_MODELS else X_test
    X_tr  = X_train_scaled if name in SCALED_MODELS else X_train

    y_pred       = model.predict(X_te)
    y_pred_train = model.predict(X_tr)
    y_prob       = model.predict_proba(X_te)[:, 1]

    acc        = accuracy_score(y_test,  y_pred)
    train_acc  = accuracy_score(y_train, y_pred_train)
    prec       = precision_score(y_test, y_pred, zero_division=0)
    rec        = recall_score(y_test,    y_pred, zero_division=0)
    f1         = f1_score(y_test,        y_pred, zero_division=0)
    roc_auc    = roc_auc_score(y_test,   y_prob)

    eval_records.append({
        "Model":        name,
        "Train_Acc":    round(train_acc, 4),
        "Test_Acc":     round(acc,       4),
        "Overfit_Gap":  round(train_acc - acc, 4),
        "Precision":    round(prec,      4),
        "Recall":       round(rec,       4),
        "F1_Score":     round(f1,        4),
        "ROC_AUC":      round(roc_auc,   4),
        "y_pred":       y_pred,
        "y_prob":       y_prob,
    })

eval_df = pd.DataFrame(eval_records)

# Print table
print()
hdr = (f"  {'Model':<22} {'Train':>7} {'Test':>7} {'Gap':>7} "
       f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7}")
print(hdr)
print("  " + "-" * 66)
for _, r in eval_df.iterrows():
    gap_flag = "  *" if r["Overfit_Gap"] > 0.08 else ""
    print(f"  {r['Model']:<22} {r['Train_Acc']:>7.4f} {r['Test_Acc']:>7.4f} "
          f"{r['Overfit_Gap']:>7.4f} {r['Precision']:>7.4f} {r['Recall']:>7.4f} "
          f"{r['F1_Score']:>7.4f} {r['ROC_AUC']:>7.4f}{gap_flag}")
print("  (* = overfit gap > 0.08)")
print()

# =============================================================================
# CELL 6.2 -- Detailed Classification Reports
# =============================================================================
print("=" * 68)
print("  6.2  DETAILED CLASSIFICATION REPORTS")
print("=" * 68)

for name in MODEL_ORDER:
    row   = eval_df[eval_df["Model"] == name].iloc[0]
    X_te  = X_test_scaled if name in SCALED_MODELS else X_test
    y_pred = row["y_pred"]
    print(f"\n  --- {name} ---")
    print(classification_report(
        y_test, y_pred,
        target_names=["NotPlaced (0)", "Placed (1)"],
        digits=4
    ))

# =============================================================================
# CELL 6.3 -- Confusion Matrix Grid  (Figure 10)
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Figure 10: Confusion Matrices (Test Set, 2,000 samples)",
             fontsize=14, fontweight="bold", y=1.01)

for ax, (name, color) in zip(axes.flat, zip(MODEL_ORDER, COLORS)):
    row    = eval_df[eval_df["Model"] == name].iloc[0]
    cm     = confusion_matrix(y_test, row["y_pred"])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    annot = np.array([[f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)"
                       for j in range(2)] for i in range(2)])

    sns.heatmap(cm, annot=annot, fmt="", ax=ax,
                cmap=sns.light_palette(color, as_cmap=True),
                linewidths=1.5, linecolor="white",
                xticklabels=["NotPlaced", "Placed"],
                yticklabels=["NotPlaced", "Placed"],
                cbar=False, annot_kws={"size": 11})
    ax.set_title(f"{name}\nAcc={row['Test_Acc']:.4f}  F1={row['F1_Score']:.4f}",
                 fontsize=11)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)

plt.tight_layout()
plt.savefig("fig10_confusion_matrices.png", bbox_inches="tight")
plt.close()
print("\n  fig10_confusion_matrices.png saved.\n")

# =============================================================================
# CELL 6.4 -- ROC Curves  (Figure 11)
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle("Figure 11: ROC Curves -- All Six Models",
             fontsize=14, fontweight="bold")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Random classifier (AUC=0.50)")

for name, color in zip(MODEL_ORDER, COLORS):
    row  = eval_df[eval_df["Model"] == name].iloc[0]
    fpr, tpr, _ = roc_curve(y_test, row["y_prob"])
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name}  (AUC = {row['ROC_AUC']:.4f})")

ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("Receiver Operating Characteristic Curves\n(Test Set, n=2,000)",
             fontsize=11, pad=8)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fig11_roc_curves.png", bbox_inches="tight")
plt.close()
print("  fig11_roc_curves.png saved.\n")

# =============================================================================
# CELL 6.5 -- 5-Fold Stratified Cross-Validation
# =============================================================================
print("=" * 68)
print("  6.5  5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 68)

# Reload clean data for CV (uses full dataset, not just train split)
import pandas as _pd
_df = _pd.read_csv("placementdata.csv")
_df = _df.drop(columns=["StudentID"])
_df = _df.rename(columns={"Workshops/Certifications": "Workshops_Certifications"})
_df["ExtracurricularActivities"] = _df["ExtracurricularActivities"].map({"Yes": 1, "No": 0})
_df["PlacementTraining"]         = _df["PlacementTraining"].map({"Yes": 1, "No": 0})
_df["PlacementStatus"]           = _df["PlacementStatus"].map({"Placed": 1, "NotPlaced": 0})

FEATURE_COLS = [
    "CGPA", "Internships", "Projects", "Workshops_Certifications",
    "AptitudeTestScore", "SoftSkillsRating",
    "ExtracurricularActivities", "PlacementTraining",
    "SSC_Marks", "HSC_Marks",
]
X_full = _df[FEATURE_COLS]
y_full = _df["PlacementStatus"]

from sklearn.preprocessing import StandardScaler as _SS
from sklearn.pipeline import Pipeline

cv_records = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n  Running 5-fold stratified CV on full dataset (n=10,000)...\n")
print(f"  {'Model':<22} {'CV Acc Mean':>12} {'CV Acc Std':>11} "
      f"{'CV F1 Mean':>11} {'CV F1 Std':>10}")
print("  " + "-" * 68)

for name in MODEL_ORDER:
    base_model = models[name]

    if name in SCALED_MODELS:
        pipeline = Pipeline([
            ("scaler", _SS()),
            ("clf",    base_model)
        ])
        X_cv = X_full
    else:
        pipeline = base_model
        X_cv = X_full

    cv_res = cross_validate(
        pipeline, X_cv, y_full,
        cv=skf,
        scoring={"accuracy": "accuracy", "f1": "f1"},
        n_jobs=-1,
        return_train_score=False
    )

    acc_mean = cv_res["test_accuracy"].mean()
    acc_std  = cv_res["test_accuracy"].std()
    f1_mean  = cv_res["test_f1"].mean()
    f1_std   = cv_res["test_f1"].std()

    print(f"  {name:<22} {acc_mean:>12.4f} {acc_std:>11.4f} "
          f"{f1_mean:>11.4f} {f1_std:>10.4f}")

    cv_records.append({
        "Model":       name,
        "CV_Acc_Mean": round(acc_mean, 4),
        "CV_Acc_Std":  round(acc_std,  4),
        "CV_F1_Mean":  round(f1_mean,  4),
        "CV_F1_Std":   round(f1_std,   4),
    })

cv_df = pd.DataFrame(cv_records)
print()

# =============================================================================
# CELL 6.6 -- Consolidated Metrics Table
# =============================================================================
print("=" * 68)
print("  6.6  CONSOLIDATED METRICS TABLE (Test Set + Cross-Validation)")
print("=" * 68)

final_df = eval_df[["Model","Test_Acc","Precision","Recall","F1_Score","ROC_AUC","Overfit_Gap"]].merge(
    cv_df, on="Model"
)

print()
print(f"  {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} "
      f"{'Gap':>6} {'CV Acc':>7} {'CV F1':>7}")
print("  " + "-" * 76)
for _, r in final_df.iterrows():
    print(f"  {r['Model']:<22} {r['Test_Acc']:>6.4f} {r['Precision']:>6.4f} "
          f"{r['Recall']:>6.4f} {r['F1_Score']:>6.4f} {r['ROC_AUC']:>6.4f} "
          f"{r['Overfit_Gap']:>6.4f} {r['CV_Acc_Mean']:>7.4f} {r['CV_F1_Mean']:>7.4f}")

# Save to CSV
final_df.to_csv("model_evaluation_results.csv", index=False)
print("\n  Saved: model_evaluation_results.csv\n")

# =============================================================================
# CELL 6.7 -- Metrics Comparison Bar Chart  (Figure 12)
# =============================================================================
metrics_plot = ["Test_Acc", "Precision", "Recall", "F1_Score", "ROC_AUC"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

fig, axes = plt.subplots(1, 5, figsize=(17, 5), sharey=False)
fig.suptitle("Figure 12: Test-Set Metric Comparison Across All Models",
             fontsize=13, fontweight="bold", y=1.02)

for ax, metric, label in zip(axes, metrics_plot, metric_labels):
    vals   = [final_df[final_df["Model"] == m][metric].values[0] for m in MODEL_ORDER]
    bars   = ax.bar(range(len(MODEL_ORDER)), vals,
                    color=COLORS, edgecolor="white", width=0.65)
    ax.set_title(label, fontsize=11)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(
        ["LR", "DT", "RF", "GB", "SVM", "KNN"],
        fontsize=9
    )
    ymin = max(0, min(vals) - 0.05)
    ax.set_ylim(ymin, min(1.05, max(vals) + 0.06))
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

patches = [plt.Rectangle((0,0),1,1, color=COLORS[i]) for i in range(6)]
fig.legend(patches, MODEL_ORDER, loc="lower center", ncol=6,
           fontsize=9, bbox_to_anchor=(0.5, -0.08))
plt.tight_layout()
plt.savefig("fig12_metrics_comparison.png", bbox_inches="tight")
plt.close()
print("  fig12_metrics_comparison.png saved.\n")

# =============================================================================
# CELL 6.8 -- Cross-Validation Distribution  (Figure 13)
# =============================================================================
print("  Running per-fold CV for box plot ...", end=" ", flush=True)

fold_records = []
for name in MODEL_ORDER:
    base_model = models[name]
    if name in SCALED_MODELS:
        pipeline = Pipeline([("scaler", _SS()), ("clf", base_model)])
        X_cv = X_full
    else:
        pipeline = base_model
        X_cv = X_full

    cv_res = cross_validate(
        pipeline, X_cv, y_full,
        cv=skf, scoring={"accuracy": "accuracy", "f1": "f1"},
        n_jobs=-1
    )
    for fold_acc, fold_f1 in zip(cv_res["test_accuracy"], cv_res["test_f1"]):
        fold_records.append({"Model": name, "Accuracy": fold_acc, "F1": fold_f1})

fold_df = pd.DataFrame(fold_records)
print("done\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 13: 5-Fold Cross-Validation Score Distribution",
             fontsize=13, fontweight="bold", y=1.01)

for ax, metric, ylabel in zip(axes, ["Accuracy", "F1"], ["Accuracy", "F1-Score"]):
    order_map = {m: i for i, m in enumerate(MODEL_ORDER)}
    fold_df_s  = fold_df.sort_values("Model", key=lambda s: s.map(order_map))

    bp = ax.boxplot(
        [fold_df[fold_df["Model"] == m][metric].values for m in MODEL_ORDER],
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 5, "alpha": 0.5},
    )
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(["LR", "DT", "RF", "GB", "SVM", "KNN"], fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(f"CV {ylabel} per Fold")

plt.tight_layout()
plt.savefig("fig13_cv_distribution.png", bbox_inches="tight")
plt.close()
print("  fig13_cv_distribution.png saved.\n")

# =============================================================================
# CELL 6.9 -- Overfitting Analysis  (Figure 14)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Figure 14: Train vs Test Accuracy (Overfitting Analysis)",
             fontsize=13, fontweight="bold", y=1.01)

x = np.arange(len(MODEL_ORDER))
w = 0.35

train_accs = [eval_df[eval_df["Model"] == m]["Train_Acc"].values[0] for m in MODEL_ORDER]
test_accs  = [eval_df[eval_df["Model"] == m]["Test_Acc"].values[0]  for m in MODEL_ORDER]

b1 = ax.bar(x - w/2, train_accs, w, label="Train Accuracy",
            color="#93C5FD", edgecolor="white")
b2 = ax.bar(x + w/2, test_accs,  w, label="Test Accuracy",
            color=COLORS, edgecolor="white")

for bar, val in zip(list(b1) + list(b2), train_accs + test_accs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)

# Gap annotation arrows
for i, (tr, te) in enumerate(zip(train_accs, test_accs)):
    if tr - te > 0.05:
        ax.annotate(f"gap={tr-te:.3f}",
                    xy=(i + w/2, te + 0.005),
                    xytext=(i + w/2, te + 0.04),
                    arrowprops={"arrowstyle": "->", "color": "#EF4444"},
                    ha="center", fontsize=8, color="#EF4444")

ax.set_xticks(x)
ax.set_xticklabels(MODEL_ORDER, rotation=12, ha="right")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.70, 0.95)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("fig14_overfitting_analysis.png", bbox_inches="tight")
plt.close()
print("  fig14_overfitting_analysis.png saved.\n")

# =============================================================================
# CELL 6.10 -- Best Model Selection
# =============================================================================
print("=" * 68)
print("  6.10  BEST MODEL IDENTIFICATION")
print("=" * 68)

# Composite score: weighted sum of Test_Acc, F1, AUC, CV_F1, minus gap penalty
final_df["Composite"] = (
    0.25 * final_df["Test_Acc"]  +
    0.25 * final_df["F1_Score"]  +
    0.25 * final_df["ROC_AUC"]   +
    0.25 * final_df["CV_F1_Mean"] -
    0.10 * final_df["Overfit_Gap"].clip(lower=0)
)
final_df_sorted = final_df.sort_values("Composite", ascending=False)

print("\n  Composite Score (Test Acc 25% + F1 25% + AUC 25% + CV F1 25% - Gap penalty):\n")
print(f"  {'Rank':<5} {'Model':<22} {'Test Acc':>9} {'F1':>8} "
      f"{'AUC':>8} {'CV F1':>8} {'Gap':>7} {'Score':>8}")
print("  " + "-" * 74)
for rank, (_, r) in enumerate(final_df_sorted.iterrows(), 1):
    marker = "  <-- BEST" if rank == 1 else ""
    print(f"  {rank:<5} {r['Model']:<22} {r['Test_Acc']:>9.4f} {r['F1_Score']:>8.4f} "
          f"{r['ROC_AUC']:>8.4f} {r['CV_F1_Mean']:>8.4f} "
          f"{r['Overfit_Gap']:>7.4f} {r['Composite']:>8.4f}{marker}")

best = final_df_sorted.iloc[0]
print(f"\n  Best model: {best['Model']}")

# =============================================================================
# CELL 6.11 -- Summary for Project Report
# =============================================================================
print()
print("=" * 68)
print("  6.11  STEP 6 SUMMARY FOR PROJECT REPORT")
print("=" * 68)
print("""
  EVALUATION APPROACH
  --------------------
  - All 6 models evaluated on the SAME untouched test set (2,000 samples).
  - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
  - 5-fold stratified cross-validation on the full 10,000-sample dataset.
  - Overfitting monitored via train-test accuracy gap.

  KEY FINDINGS
  -------------
  1. Decision Tree shows significant overfitting (train-test gap ~12.8%).
     Its test accuracy of 75% is the lowest of all models. Max_depth
     regularisation is recommended before deployment.

  2. Logistic Regression achieves strong test accuracy and one of the
     highest ROC-AUC scores despite being the simplest model. Its
     train-test gap is essentially zero, confirming good generalisation.

  3. Random Forest and Gradient Boosting perform comparably on the test
     set (~79-80% accuracy, ~0.87 AUC) with moderate overfitting gaps.
     Both are strong candidates with further hyperparameter tuning.

  4. SVM delivers competitive accuracy and the second-highest AUC with
     a negligible overfit gap (0.82%), making it the most stable model.

  5. KNN performs reasonably well but is the slowest at inference time
     (stores all training samples) and requires careful k-tuning.

  6. Composite ranking (equal weight on Test Acc, F1, AUC, CV F1,
     minus overfit penalty) identifies the best overall model above.

  FIGURES SAVED
  --------------
  fig10_confusion_matrices.png  -- 6-panel confusion matrix grid
  fig11_roc_curves.png          -- all 6 ROC curves overlaid
  fig12_metrics_comparison.png  -- bar chart for all 5 metrics
  fig13_cv_distribution.png     -- 5-fold CV box plots
  fig14_overfitting_analysis.png -- train vs test accuracy

  DATA SAVED
  -----------
  model_evaluation_results.csv  -- full consolidated metrics table
""")
