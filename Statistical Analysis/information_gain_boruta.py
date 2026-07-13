#same script can be used for boruta and ridge regression information gain, just change data

#######-----------------IMPORTS-----------------------------
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt


###----------- Information Gain-----------------------
data = pd.read_csv("ridge_cohort.csv").drop(columns=["stay_id", "hadm_id", "subject_id"])
X = data.drop(columns=["delirium_label"])
y = data["delirium_label"]

ig = mutual_info_classif(X, y, discrete_features='auto', random_state=42)

out = (
    pd.Series(ig, index=X.columns, name="information_gain")
    .sort_values(ascending=False)
    .rename_axis("feature")
    .reset_index()
)

print(out.to_string(index=False))
out.to_csv("information_gain_ridge.csv", index=False)
######-----------Visualization-----------
TOP_N = 50
CSV = "C:/Users/cabar/OneDrive - UWSP/REU/Mimic/Features/information_gain_ridge.csv"

df = pd.read_csv(CSV)
df["information_gain"] = pd.to_numeric(df["information_gain"], errors="raise")

top = (df.nlargest(TOP_N, "information_gain")
         .sort_values("information_gain"))   # ascending so largest lands at top

names = top["feature"].tolist()
vals = top["information_gain"].tolist()
y = range(len(top))
vmax = max(vals)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.0,
})

fig, ax = plt.subplots(figsize=(8.5, 0.25 * len(top) + 1.2))

ax.hlines(y=y, xmin=0, xmax=vals, color="0.75", linewidth=1.1, zorder=2)
ax.scatter(vals, y, s=32, color="black", zorder=3)

for yi, v in zip(y, vals):
    ax.text(v + vmax * 0.015, yi, f"{v:.4f}",
            va="center", ha="left", fontsize=7.5, color="0.45")

ax.set_yticks(list(y))
ax.set_yticklabels(names, fontsize=8.5)
ax.tick_params(axis="y", length=0, pad=2)
ax.set_xlabel("Information Gain", fontsize=12)
ax.set_xlim(0, vmax * 1.20)
ax.set_ylim(-0.6, len(top) - 0.4)
ax.grid(True, axis="x", color="0.93", linewidth=0.7)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("C:/Users/cabar/OneDrive - UWSP/REU/Mimic/Features/information_gain_ridge.png",
            dpi=300, bbox_inches="tight")
print("saved")