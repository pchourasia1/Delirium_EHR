"""
SHAP analysis for delirium prediction.
Outputs:
  shap_ridge_importance.png / .pdf  - mean(|SHAP|) bar chart, top N features
  shap_ridge_beeswarm.png           - per-sample SHAP distribution, same top N
  shap_ridge_importance.csv         - full ranked importance table
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

TOP_N = 50

# ---- load ---------------------------------------------------------------
df = pd.read_csv("ridge_cohort.csv").drop(columns=["stay_id", "hadm_id", "subject_id"])
y = df["delirium_label"].astype(int)
X = df.drop(columns=["delirium_label"]).select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# ---- model (XGBoost: TreeExplainer is exact + fast) ---------------------
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
sv = explainer(X_test)                      # Explanation, shape (n, n_features)

imp_all = pd.Series(np.abs(sv.values).mean(axis=0), index=X_test.columns)
imp_all.sort_values(ascending=False).rename_axis("feature").rename(
    "mean_abs_shap").to_csv("shap_ridge_importance.csv")

# ---- select top N and subset the Explanation ----------------------------
top_feats = imp_all.nlargest(TOP_N).index
top_idx = [X_test.columns.get_loc(f) for f in top_feats]
sv_top = sv[:, top_idx]                     # keeps values, data, base_values aligned
print(f"showing top {len(top_feats)} of {X.shape[1]} features")

importance = imp_all[top_feats].sort_values(ascending=True)   # ascending for barh

# ---- bar chart (paper style) -------------------------------------------
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
fig, ax = plt.subplots(figsize=(9, 0.28 * len(importance) + 1.2))
ax.barh(range(len(importance)), importance.values, color="#1f9bff", edgecolor="none")
ax.set_yticks(range(len(importance)))
ax.set_yticklabels(importance.index, fontsize=9)
ax.tick_params(axis="y", length=0, pad=2)
ax.set_ylim(-0.6, len(importance) - 0.4)
ax.set_xlabel("mean(|SHAP value|)",
              fontsize=12)
ax.set_title(f"SHAP feature importance, top {TOP_N}", fontsize=14, pad=10)
ax.grid(True, axis="x", color="0.9")
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("shap_ridge_importance.png", dpi=300, bbox_inches="tight")
fig.savefig("shap_ridge_importance.pdf", bbox_inches="tight")
plt.close(fig)
print("saved shap_ridge_importance.png / .pdf")

# ---- ranked table -------------------------------------------------------
print(f"\nMean |SHAP| importance, top {TOP_N} (descending):")
print(imp_all[top_feats].to_string())

# ---- beeswarm on the same top N -----------------------------------------
plt.figure()
shap.plots.beeswarm(
    sv_top,
    max_display=TOP_N,                      # == n features in sv_top, so no "sum of others" row
    plot_size=(10, 0.30 * TOP_N + 1.5),
    show=False,
)
plt.tight_layout()
plt.savefig("shap_ridge_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()
print("saved shap_ridge_beeswarm.png")