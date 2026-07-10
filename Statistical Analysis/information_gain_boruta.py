#######-----------------IMPORTS-----------------------------
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt


###----------- Information Gain-----------------------
data = pd.read_csv("boruta_cohort.csv").drop(columns=["stay_id", "hadm_id", "subject_id"])
X = data.drop(columns=["delirium_label"])
y = data["delirium_label"]

ig = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
result = pd.Series(ig, index=X.columns).sort_values(ascending=False)
print(result)

######-----------Visualization-----------

data = [
    ("lods_pulmonary", 0.027253),
    ("sepsis3", 0.017346),
    ("rass_last", 0.014984),
    ("apsiii", 0.013725),
    ("gcs_verbal", 0.013470),
    ("elix_count", 0.012056),
    ("infection", 0.011105),
    ("fed", 0.010390),
    ("vw_score", 0.010136),
    ("sapsii", 0.009849),
    ("coag", 0.006830),
    ("aki", 0.006317),
    ("ond", 0.005868),
    ("aniongap_last", 0.005497),
    ("meld", 0.003610),
    ("aniongap_mean", 0.003573),
    ("rbc_last", 0.003423),
    ("aniongap_max", 0.003333),
    ("spo2_mean", 0.003074),
    ("sirs_temp_score", 0.002737),
    ("invasive_vent_24h", 0.002443),
    ("wloss", 0.002330),
    ("admtype_OBSERVATION ADMIT", 0.000566),
    ("temperature_min", 0.000343),
    ("bun_cr_ratio", 0.000000),
    ("careunit_Neuro Stepdown", 0.000000),
]

# sort ascending so the largest sits at the top of the axis
data = sorted(data, key=lambda d: d[1])
names = [d[0] for d in data]
vals = [d[1] for d in data]
y = range(len(data))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.0,
})

fig, ax = plt.subplots(figsize=(9, 9))

ax.hlines(y=y, xmin=0, xmax=vals, color="0.6", linewidth=1.4, zorder=2)
ax.scatter(vals, y, s=55, color="black", zorder=3)

# value at the tip of each stem
for yi, v in zip(y, vals):
    ax.text(v + max(vals) * 0.012, yi, f"{v:.4f}",
            va="center", ha="left", fontsize=9, color="0.35")

ax.set_yticks(list(y))
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel("Information Gain", fontsize=12)
ax.set_xlim(0, max(vals) * 1.18)
ax.set_ylim(-0.7, len(data) - 0.3)
ax.grid(True, axis="x", color="0.95", linewidth=0.7)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("C:/Users/cabar/OneDrive - UWSP/REU/Mimic/Features/information_gain_boruta.png", dpi=300, bbox_inches="tight")
print("saved")