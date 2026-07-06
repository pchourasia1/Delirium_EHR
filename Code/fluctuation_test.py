"""
Isolation test: does the temporal fluctuation signal add anything beyond the
static features, in the best case (classical ML, no graph, no message passing)?

Three feature sets on the SAME stratified 70/15/15 split as the GNN, each scored
with a linear model and gradient boosting, reporting ROC-AUC AND PR-AUC:

    static only       23 Boruta features   (reproduces the ~0.886 classical ceiling)
    fluctuation only  96 temporal features
    static + fluct    119

Read it like this:
  * static+fluct ROC-AUC ~= static ROC-AUC  -> fluctuation is redundant for ranking.
  * BUT check PR-AUC. At 4.85% prevalence ROC-AUC is dominated by the 95% negatives
    and can be flat while the model's ability to surface true positives changes.
    If static+fluct PR-AUC > static PR-AUC, the features DO help on the metric that
    matters for rare-event detection -- a real result the GNN's ROC-AUC hid.
  * fluctuation-only tells you whether the temporal features carry standalone
    delirium signal at all (near 0.5 = essentially none; well above = real but
    possibly already captured by the static severity scores).

Needs: boruta_cohort.csv, patient_fluctuation.csv.gz  (sklearn only, no torch)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 42
cohort = pd.read_csv('boruta_cohort.csv')
fluct = pd.read_csv('patient_fluctuation.csv.gz')

ID_COLS = ['stay_id', 'subject_id', 'hadm_id']
LABEL_COL = 'delirium_label'
static_cols = [c for c in cohort.columns if c not in ID_COLS + [LABEL_COL]]
fluct_cols = [c for c in fluct.columns if c != 'stay_id']

df = cohort.merge(fluct, on='stay_id', how='left')
y = df[LABEL_COL].values.astype(int)
print(f'patients: {len(df)}   positives: {y.sum()} ({y.mean()*100:.2f}%)   '
      f'static: {len(static_cols)}   fluct: {len(fluct_cols)}')

# EXACT same two-stage split as the GNN (seed 42): train = 70%, test = the 15% test fold
idx = np.arange(len(y))
train_idx, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
val_idx, test_idx = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)


def score(cols):
    X = df[cols].values.astype(np.float32)
    # impute NaN with train-population mean, then standardize (fit on train only)
    cm = np.nanmean(X[train_idx], axis=0)
    cm = np.where(np.isnan(cm), 0.0, cm)
    r, c = np.where(np.isnan(X))
    X[r, c] = np.take(cm, c)
    X = StandardScaler().fit(X[train_idx]).transform(X)
    out = {}
    models = [
        ('logreg', LogisticRegression(max_iter=3000, class_weight='balanced')),
        ('histgb', HistGradientBoostingClassifier(random_state=SEED,
                                                  class_weight='balanced')),
    ]
    for name, clf in models:
        clf.fit(X[train_idx], y[train_idx])
        p = clf.predict_proba(X[test_idx])[:, 1]
        out[name] = (roc_auc_score(y[test_idx], p),
                     average_precision_score(y[test_idx], p))
    return out


sets = [('static only', static_cols),
        ('fluctuation only', fluct_cols),
        ('static + fluct', static_cols + fluct_cols)]

print(f'\n{"feature set":18s} {"model":8s} {"ROC-AUC":>8s} {"PR-AUC":>8s}')
print('-' * 46)
results = {}
for name, cols in sets:
    r = score(cols)
    results[name] = r
    for m, (auc, ap) in r.items():
        print(f'{name:18s} {m:8s} {auc:8.4f} {ap:8.4f}')

# headline deltas (histgb, usually the stronger model)
base_auc, base_ap = results['static only']['histgb']
comb_auc, comb_ap = results['static + fluct']['histgb']
print('\nfluctuation contribution on top of static (histgb):')
print(f'  ROC-AUC: {base_auc:.4f} -> {comb_auc:.4f}  (delta {comb_auc-base_auc:+.4f})')
print(f'  PR-AUC : {base_ap:.4f} -> {comb_ap:.4f}  (delta {comb_ap-base_ap:+.4f})')
print('\nPR-AUC baseline (predicting prevalence) =', round(y[test_idx].mean(), 4))