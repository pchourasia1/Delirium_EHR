import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list

df = pd.read_csv("ridge_cohort.csv").drop(columns=["stay_id", "hadm_id", "subject_id"])

target = None
if "delirium_label" in df.columns:
    target = df["delirium_label"].copy()
    df = df.drop(columns=["delirium_label"])

# Pearson works on numeric columns only
df = df.select_dtypes(include=[np.number])

corr = df.corr(method="pearson")

# order columns by hierarchical clustering of (1 - |corr|)
if corr.shape[0] > 2:
    dist = 1 - corr.abs()
    link = linkage(dist.values[np.triu_indices_from(dist, k=1)], method="average")
    order = leaves_list(link)
    corr = corr.iloc[order, order]

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

# ---- heatmap (lower triangle only) --------------------------------------
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
n = corr.shape[0]
fig, ax = plt.subplots(figsize=(max(9, n * 0.5), max(8, n * 0.5)))

sns.heatmap(
    corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.5, linecolor="white",
    annot=True, fmt=".2f", annot_kws={"size": 7},
    cbar_kws={"shrink": 0.6, "label": "Pearson r"}, ax=ax,
)
ax.set_title("Pearson correlation of features", fontsize=15, pad=12)
plt.xticks(rotation=90, fontsize=9)
plt.yticks(rotation=0, fontsize=9)
fig.tight_layout()
fig.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
fig.savefig("correlation_heatmap.pdf", bbox_inches="tight")
print("saved correlation_heatmap.png / .pdf")

# ---- report highly correlated pairs -------------------------------------
pairs = (
    corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        .stack()
        .rename("r")
        .reset_index()
)
pairs.columns = ["feature_a", "feature_b", "r"]
redundant = pairs[pairs["r"].abs() >= 0.80].sort_values("r", key=abs, ascending=False)
print("\nFeature pairs with |r| >= 0.80:")
print(redundant.to_string(index=False) if len(redundant) else "  none")

# ---- correlation with the target ----------------------------------------
if target is not None:
    tcorr = (
        df.apply(lambda c: c.corr(target, method="pearson"))
          .dropna()
          .sort_values(key=abs, ascending=True)
    )
    figt, axt = plt.subplots(figsize=(8, max(6, len(tcorr) * 0.3)))
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in tcorr.values]
    axt.hlines(range(len(tcorr)), 0, tcorr.values, color="0.6", lw=1.3, zorder=2)
    axt.scatter(tcorr.values, range(len(tcorr)), color=colors, s=45, zorder=3)
    axt.set_yticks(range(len(tcorr)))
    axt.set_yticklabels(tcorr.index, fontsize=10)
    axt.axvline(0, color="black", lw=0.8)
    axt.set_xlabel("Pearson r with delirium_label", fontsize=13)
    axt.grid(True, axis="x", color="0.9")
    axt.set_axisbelow(True)
    for s in ("top", "right"):
        axt.spines[s].set_visible(False)
    figt.tight_layout()
    figt.savefig("correlation_with_target_pearson.png", dpi=300, bbox_inches="tight")
    print("saved correlation_with_target_pearson.png")

