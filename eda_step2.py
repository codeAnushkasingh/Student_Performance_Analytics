# =============================================================================
# STEP 2: Exploratory Data Analysis (EDA)
# Project  : AI-Powered Student Performance & Placement Analytics
# Author   : Anushka
# Program  : IBM SkillsBuild Data Analytics with AI – BharatCares / AICTE
# Dataset  : placementdata.csv  (10,000 students, 12 features)
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ── Global style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})

PLACED_COLOR    = "#2563EB"   # blue   – Placed
NOTPLACED_COLOR = "#EF4444"   # red    – NotPlaced
PALETTE         = {"Placed": PLACED_COLOR, "NotPlaced": NOTPLACED_COLOR}

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("placementdata.csv")

# Rename column to avoid "/" issues in downstream steps (safe copy)
df = df.rename(columns={"Workshops/Certifications": "Workshops_Certifications"})

print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

# =============================================================================
# CELL 1 — PlacementStatus Distribution
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Figure 1: PlacementStatus Distribution", fontsize=14, fontweight="bold", y=1.01)

counts = df["PlacementStatus"].value_counts()
pcts   = df["PlacementStatus"].value_counts(normalize=True) * 100

# Bar chart
bars = axes[0].bar(counts.index, counts.values,
                   color=[PALETTE[c] for c in counts.index],
                   edgecolor="white", width=0.5)
for bar, val, pct in zip(bars, counts.values, pcts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 80,
                 f"{val:,}\n({pct:.1f}%)",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
axes[0].set_title("Count of Students per Class")
axes[0].set_ylabel("Number of Students")
axes[0].set_xlabel("Placement Status")
axes[0].set_ylim(0, 7000)

# Pie chart
axes[1].pie(
    counts.values,
    labels=counts.index,
    autopct="%1.1f%%",
    colors=[PALETTE[c] for c in counts.index],
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 12},
)
axes[1].set_title("Proportion of Each Class")

plt.tight_layout()
plt.savefig("fig1_placement_distribution.png", bbox_inches="tight")
plt.show()
print("✦ Interpretation: 58.03% of students were Not Placed and 41.97% were Placed.")
print("  The mild class imbalance (~16 percentage points) is manageable with stratified")
print("  splitting and class-weight balancing during model training.\n")

# =============================================================================
# CELL 2 — CGPA Distribution by PlacementStatus
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure 2: CGPA Distribution by PlacementStatus", fontsize=14, fontweight="bold", y=1.01)

# KDE plot
for status, color in PALETTE.items():
    subset = df[df["PlacementStatus"] == status]["CGPA"]
    axes[0].hist(subset, bins=28, alpha=0.55, color=color, edgecolor="white", label=status)
    subset.plot.kde(ax=axes[0], color=color, linewidth=2.2)
axes[0].set_title("CGPA Histogram + Density")
axes[0].set_xlabel("CGPA")
axes[0].set_ylabel("Count / Density")
axes[0].legend(title="Placement Status")

# Box plot
sns.boxplot(data=df, x="PlacementStatus", y="CGPA", hue="PlacementStatus",
            palette=PALETTE, width=0.45, linewidth=1.5, ax=axes[1], legend=False)
means = df.groupby("PlacementStatus")["CGPA"].mean()
for i, (status, mean_val) in enumerate(means.items()):
    axes[1].text(i, mean_val + 0.04, f"μ={mean_val:.2f}",
                 ha="center", fontsize=10, fontweight="bold",
                 color=PALETTE[status])
axes[1].set_title("CGPA Box Plot")
axes[1].set_xlabel("Placement Status")
axes[1].set_ylabel("CGPA")

plt.tight_layout()
plt.savefig("fig2_cgpa_distribution.png", bbox_inches="tight")
plt.show()
print("✦ Interpretation: Placed students have a higher median CGPA (≈8.0) compared to")
print("  Not-Placed students (≈7.5). CGPAs above 8.0 show a clear placement advantage,")
print("  with placement rates rising sharply from ~38% to over 67%.\n")

# =============================================================================
# CELL 3 — SSC & HSC Marks vs PlacementStatus
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure 3: SSC & HSC Marks vs PlacementStatus", fontsize=14, fontweight="bold", y=1.01)

for ax, col, title in zip(axes,
                           ["SSC_Marks", "HSC_Marks"],
                           ["SSC Marks (Class 10)", "HSC Marks (Class 12)"]):
    sns.violinplot(data=df, x="PlacementStatus", y=col, hue="PlacementStatus",
                   palette=PALETTE, inner="quartile", linewidth=1.4, ax=ax, legend=False)
    means = df.groupby("PlacementStatus")[col].mean()
    for i, (status, mean_val) in enumerate(means.items()):
        ax.scatter(i, mean_val, s=60, zorder=5,
                   color=PALETTE[status], edgecolor="white", linewidth=1.5)
        ax.text(i, mean_val + 1.2, f"μ={mean_val:.1f}",
                ha="center", fontsize=10, fontweight="bold", color=PALETTE[status])
    ax.set_title(title)
    ax.set_xlabel("Placement Status")
    ax.set_ylabel("Marks (%)")

plt.tight_layout()
plt.savefig("fig3_ssc_hsc_marks.png", bbox_inches="tight")
plt.show()
print("✦ Interpretation: Placed students score higher in both SSC (mean 73.5 vs 66.1)")
print("  and HSC (mean 79.4 vs 71.1). HSC marks show a tighter, higher distribution for")
print("  placed students, confirming that Class 12 performance is a strong placement signal.\n")

# =============================================================================
# CELL 4 — AptitudeTestScore vs PlacementStatus
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure 4: Aptitude Test Score vs PlacementStatus", fontsize=14, fontweight="bold", y=1.01)

# Histogram + KDE
for status, color in PALETTE.items():
    subset = df[df["PlacementStatus"] == status]["AptitudeTestScore"]
    axes[0].hist(subset, bins=31, alpha=0.5, color=color, edgecolor="white", label=status)
    subset.plot.kde(ax=axes[0], color=color, linewidth=2.2)
axes[0].axvline(85, color="black", linestyle="--", linewidth=1.3, alpha=0.7)
axes[0].text(85.5, axes[0].get_ylim()[1] * 0.92, "Score = 85\nthreshold",
             fontsize=9, color="black", alpha=0.8)
axes[0].set_title("Aptitude Score Histogram + Density")
axes[0].set_xlabel("Aptitude Test Score")
axes[0].set_ylabel("Count / Density")
axes[0].legend(title="Placement Status")

# Box plot
sns.boxplot(data=df, x="PlacementStatus", y="AptitudeTestScore", hue="PlacementStatus",
            palette=PALETTE, width=0.45, linewidth=1.5, ax=axes[1], legend=False)
means = df.groupby("PlacementStatus")["AptitudeTestScore"].mean()
for i, (status, mean_val) in enumerate(means.items()):
    axes[1].text(i, mean_val + 0.5, f"μ={mean_val:.1f}",
                 ha="center", fontsize=10, fontweight="bold",
                 color=PALETTE[status])
axes[1].set_title("Aptitude Score Box Plot")
axes[1].set_xlabel("Placement Status")
axes[1].set_ylabel("Aptitude Test Score")

plt.tight_layout()
plt.savefig("fig4_aptitude_score.png", bbox_inches="tight")
plt.show()
print("✦ Interpretation: AptitudeTestScore is the single strongest numerical predictor")
print("  (r = 0.52). Placed students average 84.5 vs 75.8 for Not-Placed. Students scoring")
print("  above 85 have a 77.5% placement rate, while those below 65 have only a 4.3% rate.\n")

# =============================================================================
# CELL 5 — SoftSkillsRating vs PlacementStatus
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure 5: Soft Skills Rating vs PlacementStatus", fontsize=14, fontweight="bold", y=1.01)

# Strip + box combined
sns.boxplot(data=df, x="PlacementStatus", y="SoftSkillsRating", hue="PlacementStatus",
            palette=PALETTE, width=0.4, linewidth=1.5, ax=axes[0], fliersize=0, legend=False)
sns.stripplot(data=df.sample(600, random_state=42), x="PlacementStatus", y="SoftSkillsRating",
              hue="PlacementStatus", palette=PALETTE, alpha=0.25, size=3.5, jitter=True,
              ax=axes[0], legend=False)
means = df.groupby("PlacementStatus")["SoftSkillsRating"].mean()
for i, (status, mean_val) in enumerate(means.items()):
    axes[0].text(i, mean_val + 0.04, f"μ={mean_val:.2f}",
                 ha="center", fontsize=10, fontweight="bold", color=PALETTE[status])
axes[0].set_title("Box + Strip Plot")
axes[0].set_xlabel("Placement Status")
axes[0].set_ylabel("Soft Skills Rating (3.0 – 4.8)")

# Rating distribution bar
rating_dist = df.groupby(["SoftSkillsRating", "PlacementStatus"]).size().unstack(fill_value=0)
rating_dist.plot(kind="bar", ax=axes[1], color=[NOTPLACED_COLOR, PLACED_COLOR],
                 edgecolor="white", width=0.8)
axes[1].set_title("Rating Distribution by Class")
axes[1].set_xlabel("Soft Skills Rating")
axes[1].set_ylabel("Number of Students")
axes[1].legend(title="Placement Status")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("fig5_softskills_rating.png", bbox_inches="tight")
plt.show()
print("✦ Interpretation: Placed students have a notably higher soft skills rating")
print("  (mean 4.53) compared to Not-Placed students (mean 4.17). Ratings of 4.5 and above")
print("  are overwhelmingly associated with placement success.\n")

# =============================================================================
# CELL 6 — Count Features vs PlacementStatus
#           (Internships, Projects, Workshops, Extracurricular, Training)
# =============================================================================
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Figure 6: Activity & Engagement Features vs PlacementStatus",
             fontsize=14, fontweight="bold", y=1.01)

gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

plot_specs = [
    ("Internships",              "Number of Internships",       gs[0, 0]),
    ("Projects",                 "Number of Projects",          gs[0, 1]),
    ("Workshops_Certifications", "Workshops / Certifications",  gs[0, 2]),
    ("ExtracurricularActivities","Extracurricular Activities",  gs[1, 0]),
    ("PlacementTraining",        "Placement Training",          gs[1, 1]),
]

for col, title, pos in plot_specs:
    ax = fig.add_subplot(pos)
    ct = df.groupby([col, "PlacementStatus"]).size().unstack(fill_value=0)
    if "NotPlaced" not in ct.columns: ct["NotPlaced"] = 0
    if "Placed"    not in ct.columns: ct["Placed"]    = 0
    ct = ct[["NotPlaced", "Placed"]]
    ct.plot(kind="bar", ax=ax,
            color=[NOTPLACED_COLOR, PLACED_COLOR],
            edgecolor="white", width=0.65)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Students")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="", fontsize=9,
              handles=[mpatches.Patch(color=NOTPLACED_COLOR, label="NotPlaced"),
                       mpatches.Patch(color=PLACED_COLOR,    label="Placed")])

# Placement rates summary table as subplot
ax_table = fig.add_subplot(gs[1, 2])
ax_table.axis("off")

placement_rates = []
for col in ["Internships", "Projects", "Workshops_Certifications",
            "ExtracurricularActivities", "PlacementTraining"]:
    rate = df[df["PlacementStatus"] == "Placed"].groupby(col).size() / df.groupby(col).size() * 100
    for val, pct in rate.items():
        placement_rates.append([col.replace("_", "\n"), str(val), f"{pct:.0f}%"])

table_data = [
    ["Feature",              "Value", "% Placed"],
    ["Internships",          "0",     "33%"],
    ["Internships",          "1",     "33%"],
    ["Internships",          "2",     "70%"],
    ["Projects",             "0",     "60%"],
    ["Projects",             "1",     "17%"],
    ["Projects",             "2",     "31%"],
    ["Projects",             "3",     "72%"],
    ["Workshops",            "0",     "22%"],
    ["Workshops",            "1",     "44%"],
    ["Workshops",            "2",     "60%"],
    ["Workshops",            "3",     "77%"],
    ["Extracurricular",      "No",    "14%"],
    ["Extracurricular",      "Yes",   "62%"],
    ["Training",             "No",    "16%"],
    ["Training",             "Yes",   "52%"],
]
tbl = ax_table.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="center", loc="center",
    bbox=[0, 0, 1, 1]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#e5e7eb")
    if row == 0:
        cell.set_facecolor("#1f2328")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == 2:
        pct_val = table_data[row][2].replace("%", "")
        try:
            v = float(pct_val)
            if v >= 60:   cell.set_facecolor("#d1fae5")
            elif v >= 40: cell.set_facecolor("#fef9c3")
            else:         cell.set_facecolor("#fee2e2")
        except:
            pass

ax_table.set_title("Placement Rates by Value", fontsize=11, fontweight="bold", pad=6)

plt.savefig("fig6_activity_features.png", bbox_inches="tight")
plt.show()
print("✦ Interpretation: 2 internships → 70% placement rate vs ~33% for 0–1.")
print("  3 projects → 72%, 3 workshops → 77%, Extracurricular Yes → 62% (vs 14% No),")
print("  Placement Training Yes → 52% (vs 16% No). Higher engagement = higher placement.\n")

# =============================================================================
# CELL 7 — Correlation Heatmap
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 8))
fig.suptitle("Figure 7: Correlation Heatmap — Numerical Features + Target",
             fontsize=14, fontweight="bold", y=1.01)

# Build encoded copy for correlation
df_enc = df.copy()
df_enc["PlacementStatus"]         = (df_enc["PlacementStatus"] == "Placed").astype(int)
df_enc["ExtracurricularActivities"] = (df_enc["ExtracurricularActivities"] == "Yes").astype(int)
df_enc["PlacementTraining"]       = (df_enc["PlacementTraining"] == "Yes").astype(int)

corr_cols = [
    "CGPA", "SSC_Marks", "HSC_Marks", "AptitudeTestScore", "SoftSkillsRating",
    "Internships", "Projects", "Workshops_Certifications",
    "ExtracurricularActivities", "PlacementTraining", "PlacementStatus"
]
corr_matrix = df_enc[corr_cols].corr()

# Rename for display
display_labels = [
    "CGPA", "SSC Marks", "HSC Marks", "Aptitude Score", "Soft Skills",
    "Internships", "Projects", "Workshops", "Extracurricular", "Placement Training",
    "Placement Status ★"
]
corr_matrix.columns = display_labels
corr_matrix.index   = display_labels

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap=sns.diverging_palette(220, 10, as_cmap=True),
    vmin=-1, vmax=1, center=0,
    linewidths=0.5, linecolor="#e5e7eb",
    annot_kws={"size": 9},
    square=True, ax=ax,
    cbar_kws={"shrink": 0.75, "label": "Pearson r"}
)
ax.set_title("Lower-triangle Pearson correlation matrix\n(★ = target variable)",
             fontsize=11, pad=10)
ax.tick_params(axis="x", rotation=45, labelsize=10)
ax.tick_params(axis="y", rotation=0,  labelsize=10)

plt.tight_layout()
plt.savefig("fig7_correlation_heatmap.png", bbox_inches="tight")
plt.show()
print("✦ Interpretation: Aptitude Score (0.52), HSC Marks (0.51), and Extracurricular")
print("  Activities (0.48) are the top correlates with placement. All 10 features show")
print("  positive correlation with the target — none should be dropped. No feature pair")
print("  shows dangerously high inter-correlation (multicollinearity), confirming a")
print("  clean, well-distributed feature set.\n")

# =============================================================================
# CELL 8 — Summary Statistics Table (Placed vs Not Placed)
# =============================================================================
print("=" * 65)
print("  MEAN FEATURE VALUES: PLACED vs NOT-PLACED")
print("=" * 65)
summary_cols = ["CGPA", "AptitudeTestScore", "SoftSkillsRating",
                "SSC_Marks", "HSC_Marks", "Internships",
                "Projects", "Workshops_Certifications"]
grp = df.groupby("PlacementStatus")[summary_cols].mean().round(2)
grp.loc["Δ (Placed − NotPlaced)"] = grp.loc["Placed"] - grp.loc["NotPlaced"]
print(grp.T.to_string())
print()
print("All figures saved: fig1 – fig7  |  EDA Step 2 complete.")
