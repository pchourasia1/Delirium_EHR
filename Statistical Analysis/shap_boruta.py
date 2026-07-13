"""
SHAP analysis for delirium prediction.
Outputs:
  shap_importance.png / .pdf   - mean(|SHAP|) horizontal bar chart
  shap_beeswarm.png            - per-sample SHAP distribution
  prints the ranked importance table
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ---- load ---------------------------------------------------------------
df = pd.read_csv("boruta_cohort.csv").drop(columns=["stay_id", "hadm_id", "subject_id"])
y = df["delirium_label"].astype(int)
X = df.drop(columns=["delirium_label"]).select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# ---- model (XGBoost: TreeExplainer is exact + fast) ---------------------
# scale_pos_weight balances the 4.85% positive rate
spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
model = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw, eval_metric="auc",
    random_state=42, n_jobs=-1,
)
model.fit(X_train, y_train)

# ---- SHAP ---------------------------------------------------------------
explainer = shap.TreeExplainer(model)
sv = explainer(X_test)                      # Explanation object, shape (n, n_features)

mean_abs = np.abs(sv.values).mean(axis=0)
importance = (
    pd.Series(mean_abs, index=X.columns)
      .sort_values(ascending=True)
)

# ---- bar chart (paper style) -------------------------------------------
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
fig, ax = plt.subplots(figsize=(9, max(6, len(importance) * 0.32)))
ax.barh(range(len(importance)), importance.values, color="#1f9bff", edgecolor="none")
ax.set_yticks(range(len(importance)))
ax.set_yticklabels(importance.index, fontsize=10)
ax.set_xlabel("mean(|SHAP value|)  (average impact on model output magnitude)",
              fontsize=12)
ax.grid(True, axis="x", color="0.9")
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("shap_importance.png", dpi=300, bbox_inches="tight")
fig.savefig("shap_importance.pdf", bbox_inches="tight")
print("saved shap_importance.png / .pdf")

# ---- ranked table -------------------------------------------------------
print("\nMean |SHAP| importance (descending):")
print(importance.sort_values(ascending=False).to_string())

# ---- beeswarm (optional, shows direction + spread) ----------------------
plt.figure()
shap.plots.beeswarm(sv, max_display=len(X.columns), show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=300, bbox_inches="tight")
print("saved shap_beeswarm.png")