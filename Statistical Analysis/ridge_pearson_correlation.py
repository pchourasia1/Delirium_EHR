import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list

TOP_N = 50

df = pd.read_csv("ridge_cohort.csv").drop(columns=["stay_id", "hadm_id", "subject_id"])

target = None
if "delirium_label" in df.columns:
    target = df["delirium_label"].copy()
    df = df.drop(columns="delirium_label")

df = df.select_dtypes(include=[np.number])
df = df.loc[:, df.std() > 0]          # drop constant cols -> NaN corr -> broken linkage

# ---- select top N features by |r| with the target ------------------------
tcorr_all = df.apply(lambda c: c.corr(target, method="pearson")).dropna()
top_feats = tcorr_all.abs().nlargest(TOP_N).index
df = df[top_feats]
print(f"kept {df.shape[1]} of {len(tcorr_all)} features")

corr = df.corr(method="pearson")

# ---- order columns by hierarchical clustering of (1 - |corr|) ------------
if corr.shape[0] > 2:
    dist = (1 - corr.abs()).fillna(1.0)   # NaN -> max distance, keeps linkage alive
    link = linkage(dist.values[np.triu_indices_from(dist, k=1)], method="average")
    order = leaves_list(link)
    corr = corr.iloc[order, order]

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

# ---- heatmap (lower triangle, diagonal excluded) -------------------------
mask = np.triu(np.ones_like(corr, dtype=bool), k=0)   # k=0 drops the r=1 diagonal
n = corr.shape[0]
side = min(20, max(9, n * 0.24))
fig, ax = plt.subplots(figsize=(side, side))

sns.heatmap(
    corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0,
    annot=False,
    xticklabels=1, yticklabels=1,     # 1 = label every row/col, no auto-skipping
    cbar_kws={"shrink": 0.5, "aspect": 30, "pad": 0.02, "label": "Pearson r"},
    ax=ax,
)
ax.set_title(f"Pearson correlation, top {TOP_N} features", fontsize=15, pad=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
fig.savefig("ridge_pearson_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved ridge_pearson_correlation_heatmap.png")

# ---- report highly correlated pairs --------------------------------------
pairs = (
    corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        .stack().rename("r").reset_index()
)
pairs.columns = ["feature_a", "feature_b", "r"]
redundant = pairs[pairs["r"].abs() >= 0.80].sort_values("r", key=abs, ascending=False)
print(f"\nFeature pairs with |r| >= 0.80: {len(redundant)}")
print(redundant.to_string(index=False) if len(redundant) else "  none")
redundant.to_csv("ridge_redundant_pairs.csv", index=False)

# ---- correlation with the target -----------------------------------------
tcorr = tcorr_all[top_feats].sort_values(key=abs, ascending=True)

figt, axt = plt.subplots(figsize=(8.5, 0.25 * len(tcorr) + 1.2))
colors = ["#c0392b" if v > 0 else "#2471a3" for v in tcorr.values]
axt.hlines(range(len(tcorr)), 0, tcorr.values, color="0.6", lw=1.2, zorder=2)
axt.scatter(tcorr.values, range(len(tcorr)), color=colors, s=32, zorder=3)
axt.set_yticks(range(len(tcorr)))
axt.set_yticklabels(tcorr.index, fontsize=8.5)
axt.tick_params(axis="y", length=0, pad=2)
axt.axvline(0, color="black", lw=0.8)
axt.set_xlabel("Pearson r with delirium_label", fontsize=13)
axt.set_ylim(-0.6, len(tcorr) - 0.4)
axt.grid(True, axis="x", color="0.9")
axt.set_axisbelow(True)
for s in ("top", "right"):
    axt.spines[s].set_visible(False)
figt.tight_layout()
figt.savefig("ridge_pearson_correlation.png", dpi=300, bbox_inches="tight")
plt.close(figt)
print("saved ridge_pearson_correlation.png")