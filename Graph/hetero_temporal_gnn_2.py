"""
Temporal-chain heterogeneous GNN for ICU delirium prediction, with cross-block
fluctuation features fed DIRECTLY into the patient node (PyTorch Geometric).

Change vs the previous temporal model:
    The variability signal lives in how signals CHANGE over time, not in any
    single 6h snapshot. Rather than making the GNN reconstruct that from four
    separate window nodes, we precompute per-patient fluctuation features from
    the raw timeseries and attach them to the patient node. They reach the
    classifier directly through the existing skip connection, so the signal no
    longer has to survive message passing to be used.

    This is deliberately a MINIMAL change to the temporal model: the window
    chain, belongs_to pooling, and KNN edges are all unchanged. The only
    difference is the patient node now carries [23 static + 96 fluctuation]
    features instead of 23. That keeps it a clean test: if val/test AUC moves,
    the temporal signal was real and message passing over block snapshots
    couldn't extract it; if AUC still doesn't move, the fluctuation signal is
    largely redundant with the static severity scores (a legitimate finding).

Fluctuation features (per patient, 8 per signal x 12 signals = 96):
    std, range, masd (mean abs successive diff), slope (linear trend over 24h),
    ndir (successive-difference direction changes = oscillation count), and the
    block-to-block deltas d01/d12/d23 of the block means (the literal
    "RASS delta block-to-block"). Computed over each patient's own readings, so
    no cross-patient leakage; scalers/imputation are still fit on TRAIN only.

Node types:
    window  : one per (patient, 6h block), 73 per-block features
    patient : one per stay, 23 static + 96 fluctuation = 119 features, the target

Edge types (unchanged):
    (window,  precedes/rev_precedes, window)   temporal chain
    (window,  belongs_to,           patient)   pools windows into the patient
    (patient, similar_to,           patient)   KNN similarity

Operating point:
    The hard-label threshold is chosen to CATCH >= TARGET_RECALL of real
    delirium cases with the fewest false alarms, selected on validation and
    applied to test. This is a clinical-screen framing (recall-first), not F1.

Outputs:
    gnn_predictions.csv  one row per patient: stay_id, true_label,
                         prob_delirium, pred_label (argmax), pred_label_tuned
                         (recall-targeted threshold), split.

Inputs expected in the working directory:
    boruta_cohort.csv, edges_patient_patient.csv, raw_timeseries.csv
    (cached: window_features.csv.gz, patient_fluctuation.csv.gz)

Requires: torch, torch_geometric, pandas, numpy, scikit-learn
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    precision_recall_curve,
)
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

# ── paths ──
COHORT_CSV = 'boruta_cohort.csv'
KNN_CSV = 'edges_patient_patient.csv'
RAW_TS_CSV = 'raw_timeseries.csv'
WINDOW_CACHE = 'window_features.csv.gz'
FLUCT_CACHE = 'patient_fluctuation.csv.gz'
PRED_OUT = 'gnn_predictions.csv'

# ── configuration ──
SEED = 42
N_BLOCKS = 4
BLOCK_HOURS = 6
HIDDEN_DIM = 128
LR = 0.0001
WEIGHT_DECAY = 5e-4
EPOCHS = 400
DROPOUT = 0.5
FOCAL_GAMMA = 4.0
POS_WEIGHT = 1.0
NEG_WEIGHT = 2.0

USE_FLUCTUATION = True  # set False to reproduce the previous static-patient model (ablation)

SIGNALS = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2', 'temperature',
           'gcs_eye', 'gcs_verbal', 'gcs_motor', 'rass', 'glucose_vital']
STATS = ['mean', 'min', 'max', 'std', 'count']

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── focal loss (split per class) ──
def _focal_term(logits, targets, gamma):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma) * ce


def positive_focal_loss(logits, targets, weight, gamma=2.0):
    if targets.numel() == 0:
        return logits.new_zeros(())
    return weight * _focal_term(logits, targets, gamma).mean()


def negative_focal_loss(logits, targets, weight, gamma=2.0):
    if targets.numel() == 0:
        return logits.new_zeros(())
    return weight * _focal_term(logits, targets, gamma).mean()


def combined_focal_loss(logits, targets, pos_weight, neg_weight, gamma=2.0):
    pm, nm = targets == 1, targets == 0
    return (positive_focal_loss(logits[pm], targets[pm], pos_weight, gamma)
            + negative_focal_loss(logits[nm], targets[nm], neg_weight, gamma))


# ── per-block window features (unchanged from the temporal model) ──
def extract_window_features(raw_csv, n_blocks=N_BLOCKS, block_hours=BLOCK_HOURS):
    df = pd.read_csv(raw_csv,
                     dtype={'stay_id': 'int32', 'hours_from_admit': 'float32',
                            'measurement': 'category', 'value': 'float32'},
                     usecols=['stay_id', 'hours_from_admit', 'measurement', 'value'])
    df['block'] = np.minimum((df['hours_from_admit'] // block_hours).astype('int16'), n_blocks - 1)
    agg = (df.groupby(['stay_id', 'block', 'measurement'], observed=True)['value']
             .agg(STATS).reset_index())
    last = (df.sort_values('hours_from_admit')
              .groupby(['stay_id', 'block', 'measurement'], observed=True)['value']
              .last().reset_index(name='last'))
    agg = agg.merge(last, on=['stay_id', 'block', 'measurement'])
    wide = agg.set_index(['stay_id', 'block', 'measurement'])[STATS + ['last']].unstack('measurement')
    wide.columns = [f'{meas}_{stat}' for stat, meas in wide.columns]
    wide = wide.reset_index()
    all_ids = np.sort(df['stay_id'].unique())
    grid = pd.MultiIndex.from_product([all_ids, range(n_blocks)],
                                      names=['stay_id', 'block']).to_frame(index=False)
    full = grid.merge(wide, on=['stay_id', 'block'], how='left').sort_values(['stay_id', 'block']).reset_index(drop=True)
    feat_cols = [c for c in full.columns if c not in ('stay_id', 'block')]
    zero_cols = [c for c in feat_cols if c.endswith('_count') or c.endswith('_std')]
    loc_cols = [c for c in feat_cols if c.endswith(('_mean', '_min', '_max', '_last'))]
    full[zero_cols] = full[zero_cols].fillna(0)
    full[loc_cols] = full.groupby('stay_id')[loc_cols].ffill()
    full[loc_cols] = full.groupby('stay_id')[loc_cols].bfill()
    full['block_pos'] = full['block'] / (n_blocks - 1)
    return full


# ── cross-block / trajectory fluctuation features (per patient) ──
def extract_fluctuation_features(raw_csv, n_blocks=N_BLOCKS, block_hours=BLOCK_HOURS):
    """
    Per patient, per signal: std, range, masd, slope, ndir (oscillation count),
    and block-mean deltas d01/d12/d23. All computed on the patient's own
    time-ordered readings. Returns one row per stay_id, columns signal_feature.
    """
    df = pd.read_csv(raw_csv,
                     dtype={'stay_id': 'int32', 'hours_from_admit': 'float32',
                            'measurement': 'category', 'value': 'float32'},
                     usecols=['stay_id', 'hours_from_admit', 'measurement', 'value'])
    df = df.sort_values(['stay_id', 'measurement', 'hours_from_admit']).reset_index(drop=True)
    gkey = ['stay_id', 'measurement']
    g = df.groupby(gkey, observed=True)

    base = g['value'].agg(['std', 'min', 'max'])
    base['range'] = base['max'] - base['min']
    base['std'] = base['std'].fillna(0.0)

    df['absdiff'] = g['value'].diff().abs()
    masd = df.groupby(gkey, observed=True)['absdiff'].mean().rename('masd')

    # oscillation: direction changes in successive diffs. GUARD against the
    # first-per-group NaN diff (NaN != 0 and NaN != NaN are both True in pandas,
    # which would otherwise add a phantom +1 to every patient).
    df['sd'] = np.sign(g['value'].diff())
    df['sd_prev'] = df.groupby(gkey, observed=True)['sd'].shift()
    valid = df['sd'].notna() & df['sd_prev'].notna()
    df['dirchg'] = (valid & (df['sd'] != 0) & (df['sd_prev'] != 0)
                    & (df['sd'] != df['sd_prev'])).astype('int8')
    ndir = df.groupby(gkey, observed=True)['dirchg'].sum().rename('ndir')

    # slope = cov(t, v) / var(t), vectorized; var(t)=0 (single reading) -> 0
    df['tv'] = df['hours_from_admit'] * df['value']
    df['t2'] = df['hours_from_admit'] ** 2
    g2 = df.groupby(gkey, observed=True)
    mt, mv = g2['hours_from_admit'].mean(), g2['value'].mean()
    mtv, mt2 = g2['tv'].mean(), g2['t2'].mean()
    slope = ((mtv - mt * mv) / ((mt2 - mt * mt).replace(0, np.nan))).fillna(0.0).rename('slope')

    traj = base[['std', 'range']].join([masd, slope, ndir])

    # block-mean deltas (the "delta block-to-block" features)
    df['block'] = np.minimum((df['hours_from_admit'] // block_hours).astype('int16'), n_blocks - 1)
    bmp = df.groupby(['stay_id', 'block', 'measurement'], observed=True)['value'].mean().unstack('block')
    for b in range(n_blocks):
        if b not in bmp.columns:
            bmp[b] = np.nan
    bmp = bmp[list(range(n_blocks))]
    deltas = pd.DataFrame({f'd{b}{b+1}': bmp[b + 1] - bmp[b] for b in range(n_blocks - 1)},
                          index=bmp.index)

    feat = traj.join(deltas)
    wide = feat.unstack('measurement')
    wide.columns = [f'{meas}_{stat}' for stat, meas in wide.columns]
    return wide.reset_index()


def _load_or_build(cache, builder, label):
    if os.path.exists(cache):
        print(f'Loading cached {label} from {cache}')
        return pd.read_csv(cache)
    print(f'Building {label} from {RAW_TS_CSV} ...')
    out = builder(RAW_TS_CSV)
    out.to_csv(cache, index=False, compression='gzip')
    print(f'  saved {cache}')
    return out


# ── 1. patient-level data ──
cohort = pd.read_csv(COHORT_CSV)
ID_COLS = ['stay_id', 'subject_id', 'hadm_id']
LABEL_COL = 'delirium_label'
static_cols = [c for c in cohort.columns if c not in ID_COLS + [LABEL_COL]]

stay_ids = cohort['stay_id'].values
N = len(stay_ids)
labels_np = cohort[LABEL_COL].values.astype(np.int64)
labels = torch.tensor(labels_np, dtype=torch.long)
patient_static = cohort[static_cols].values.astype(np.float32)      # (N, 23)

# fluctuation features -> aligned to cohort order, concatenated onto the patient node
if USE_FLUCTUATION:
    fluct = _load_or_build(FLUCT_CACHE, extract_fluctuation_features, 'fluctuation features')
    fluct_cols = [c for c in fluct.columns if c != 'stay_id']
    fl = pd.DataFrame({'stay_id': stay_ids}).merge(fluct, on='stay_id', how='left')
    patient_raw = np.concatenate([patient_static, fl[fluct_cols].values.astype(np.float32)], axis=1)
    print(f'Patient features: {patient_static.shape[1]} static + {len(fluct_cols)} fluctuation '
          f'= {patient_raw.shape[1]}')
else:
    patient_raw = patient_static
    print(f'Patient features: {patient_raw.shape[1]} static only (fluctuation OFF)')

# ── 2. window features + align to patient ordering ──
wf = _load_or_build(WINDOW_CACHE, extract_window_features, 'window features')
window_cols = [c for c in wf.columns if c not in ('stay_id', 'block')]
wgrid = pd.DataFrame({'stay_id': np.repeat(stay_ids, N_BLOCKS),
                      'block': np.tile(np.arange(N_BLOCKS), N)})
wgrid = wgrid.merge(wf, on=['stay_id', 'block'], how='left')
W = wgrid[window_cols].values.astype(np.float32)
print(f'Windows: {W.shape[0]}  Window features: {W.shape[1]}')

# ── 3. split (patient level, stratified) ──
indices = np.arange(N)
train_idx, temp_idx = train_test_split(indices, test_size=0.30, stratify=labels_np, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=labels_np[temp_idx], random_state=SEED)


def fit_impute_scale(mat, train_mask_rows):
    """Fill NaN with train-population column mean, then standardize (fit on train rows)."""
    col_mean = np.nanmean(mat[train_mask_rows], axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    r, c = np.where(np.isnan(mat))
    mat[r, c] = np.take(col_mean, c)
    scaler = StandardScaler().fit(mat[train_mask_rows])
    return scaler.transform(mat)


# ── 4. scale (fit on TRAIN only), both patient and window matrices ──
patient_x = torch.tensor(fit_impute_scale(patient_raw, train_idx), dtype=torch.float)

w_patient_idx = np.arange(W.shape[0]) // N_BLOCKS
train_w = np.isin(w_patient_idx, train_idx)
window_x = torch.tensor(fit_impute_scale(W, train_w), dtype=torch.float)

# ── 5. build temporal HeteroData ──
data = HeteroData()
data['patient'].x = patient_x
data['patient'].y = labels
data['window'].x = window_x

p_arr = np.repeat(np.arange(N), N_BLOCKS - 1)
b_arr = np.tile(np.arange(N_BLOCKS - 1), N)
prec_src = torch.tensor(p_arr * N_BLOCKS + b_arr, dtype=torch.long)
prec_dst = torch.tensor(p_arr * N_BLOCKS + b_arr + 1, dtype=torch.long)
data['window', 'precedes', 'window'].edge_index = torch.stack([prec_src, prec_dst])
data['window', 'rev_precedes', 'window'].edge_index = torch.stack([prec_dst, prec_src])

w_ids = torch.arange(W.shape[0], dtype=torch.long)
data['window', 'belongs_to', 'patient'].edge_index = torch.stack([w_ids, w_ids // N_BLOCKS])

knn = pd.read_csv(KNN_CSV)
pp_src = torch.tensor(knn['src_patient_id'].values, dtype=torch.long)
pp_dst = torch.tensor(knn['dst_patient_id'].values, dtype=torch.long)
data['patient', 'similar_to', 'patient'].edge_index = torch.stack([pp_src, pp_dst])

# ── 6. masks on patient nodes ──
for name, idx in [('train_mask', train_idx), ('val_mask', val_idx), ('test_mask', test_idx)]:
    m = torch.zeros(N, dtype=torch.bool)
    m[idx] = True
    data['patient'][name] = m
_tr, _va, _te = data['patient'].train_mask, data['patient'].val_mask, data['patient'].test_mask
assert (_tr & _va).sum() == 0 and (_tr & _te).sum() == 0 and (_va & _te).sum() == 0, 'mask overlap!'
print(f'Train: {int(_tr.sum())}  Val: {int(_va.sum())}  Test: {int(_te.sum())}  '
      f'Train pos rate: {labels[_tr].float().mean():.4f}')


# ── 7. model (unchanged structure; patient_in now wider) ──
class TemporalHeteroGNN(nn.Module):
    def __init__(self, window_in, patient_in, hidden, dropout):
        super().__init__()
        self.conv1 = HeteroConv({
            ('window', 'precedes', 'window'): SAGEConv(window_in, hidden),
            ('window', 'rev_precedes', 'window'): SAGEConv(window_in, hidden),
            ('window', 'belongs_to', 'patient'): SAGEConv((window_in, patient_in), hidden),
            ('patient', 'similar_to', 'patient'): SAGEConv(patient_in, hidden),
        }, aggr='sum')
        self.norm_w1 = nn.LayerNorm(hidden)
        self.norm_p1 = nn.LayerNorm(hidden)
        self.conv2 = HeteroConv({
            ('window', 'precedes', 'window'): SAGEConv(hidden, hidden),
            ('window', 'rev_precedes', 'window'): SAGEConv(hidden, hidden),
            ('window', 'belongs_to', 'patient'): SAGEConv((hidden, hidden), hidden),
            ('patient', 'similar_to', 'patient'): SAGEConv(hidden, hidden),
        }, aggr='sum')
        self.norm_p2 = nn.LayerNorm(hidden)
        # skip connection concatenates the FULL raw patient vector (static + fluctuation),
        # so the fluctuation features reach the classifier directly.
        self.classifier = nn.Sequential(
            nn.Linear(hidden + patient_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.dropout = dropout

    def forward(self, data):
        p_raw = data['patient'].x
        x = {'window': data['window'].x, 'patient': data['patient'].x}
        eid = {
            ('window', 'precedes', 'window'): data['window', 'precedes', 'window'].edge_index,
            ('window', 'rev_precedes', 'window'): data['window', 'rev_precedes', 'window'].edge_index,
            ('window', 'belongs_to', 'patient'): data['window', 'belongs_to', 'patient'].edge_index,
            ('patient', 'similar_to', 'patient'): data['patient', 'similar_to', 'patient'].edge_index,
        }
        x = self.conv1(x, eid)
        x['window'] = F.relu(self.norm_w1(x['window']))
        x['patient'] = F.relu(self.norm_p1(x['patient']))
        x = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x.items()}
        x = self.conv2(x, eid)
        p = F.relu(self.norm_p2(x['patient']))
        return self.classifier(torch.cat([p, p_raw], dim=1))


# ── 8. training setup ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
model = TemporalHeteroGNN(window_in=window_x.shape[1], patient_in=patient_x.shape[1],
                          hidden=HIDDEN_DIM, dropout=DROPOUT).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)


def criterion(logits, targets):
    return combined_focal_loss(logits, targets, POS_WEIGHT, NEG_WEIGHT, FOCAL_GAMMA)


# ── 9. train / eval ──
def train_epoch():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    m = data['patient'].train_mask
    loss = criterion(out[m], data['patient'].y[m])
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data)
    probs = F.softmax(out, dim=1)[:, 1]
    preds = out.argmax(dim=1)
    yt = data['patient'].y[mask].cpu().numpy()
    yp = preds[mask].cpu().numpy()
    ypr = probs[mask].cpu().numpy()
    return {'accuracy': accuracy_score(yt, yp), 'precision': precision_score(yt, yp, zero_division=0),
            'recall': recall_score(yt, yp, zero_division=0),
            'f1_weighted': f1_score(yt, yp, average='weighted'), 'f1_macro': f1_score(yt, yp, average='macro'),
            'roc_auc': roc_auc_score(yt, ypr), 'y_true': yt, 'y_pred': yp}


# ── 10. training loop ──
best_val_auc, best_epoch, best_state = 0.0, 0, None
print(f'\n{"Epoch":>5s}  {"Loss":>7s}  {"LR":>8s}  {"Trn AUC":>8s}  {"Val F1m":>7s}  {"Val AUC":>8s}')
print('-' * 55)
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch()
    if epoch % 10 == 0 or epoch == 1:
        trn = evaluate(data['patient'].train_mask)
        val = evaluate(data['patient'].val_mask)
        lr_now = optimizer.param_groups[0]['lr']
        marker = ''
        if val['roc_auc'] > best_val_auc:
            best_val_auc, best_epoch = val['roc_auc'], epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = '  *best'
        print(f'{epoch:5d}  {loss:7.4f}  {lr_now:8.5f}  {trn["roc_auc"]:8.4f}  '
              f'{val["f1_macro"]:7.4f}  {val["roc_auc"]:8.4f}{marker}')
print(f'\nBest val AUC: {best_val_auc:.4f} at epoch {best_epoch}')

# ── 11. test ──
model.load_state_dict(best_state)
test = evaluate(data['patient'].test_mask)
print('\n' + '=' * 50)
print('TEST SET RESULTS (best checkpoint, argmax @ 0.5)')
print('=' * 50)
for k in ['accuracy', 'precision', 'recall', 'f1_weighted', 'f1_macro', 'roc_auc']:
    print(f'{k:12s} {test[k]:.4f}')

# ── 12. per-patient predictions + recall-targeted threshold ──
# One full forward pass with the best checkpoint. The hard-label threshold is
# chosen on VALIDATION to catch >= TARGET_RECALL of real cases with the fewest
# false alarms, then applied to everyone. Tune on val, never on test.
TARGET_RECALL = 0.80

model.eval()
with torch.no_grad():
    logits = model(data)
    prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()      # P(delirium) per patient
    pred_argmax = logits.argmax(dim=1).cpu().numpy()         # argmax @ 0.5

y_all = data['patient'].y.cpu().numpy()
val_true, val_prob = y_all[val_idx], prob[val_idx]

# precision_recall_curve: prec[:-1] and rec[:-1] align with thr. Keep thresholds
# that meet the recall floor, then take the highest-precision one (fewest FPs).
prec, rec, thr = precision_recall_curve(val_true, val_prob)
meets = rec[:-1] >= TARGET_RECALL
if meets.any():
    best_th = float(thr[meets][np.argmax(prec[:-1][meets])])
else:
    best_th = float(thr.min())   # even the lowest threshold misses the floor (shouldn't happen)
pred_tuned = (prob >= best_th).astype(int)

vp = (val_prob >= best_th).astype(int)
print(f'\nthreshold for >={TARGET_RECALL:.0%} recall: {best_th:.3f}  '
      f'(val recall {recall_score(val_true, vp, zero_division=0):.3f}, '
      f'val precision {precision_score(val_true, vp, zero_division=0):.3f})')

split = np.empty(N, dtype=object)
split[train_idx], split[val_idx], split[test_idx] = 'train', 'val', 'test'
pred_df = pd.DataFrame({
    'stay_id': stay_ids,
    'true_label': y_all,
    'prob_delirium': prob,
    'pred_label': pred_argmax,          # argmax @ 0.5
    'pred_label_tuned': pred_tuned,     # recall-targeted threshold, the one to use
    'split': split,
}).sort_values('prob_delirium', ascending=False)
pred_df.to_csv(PRED_OUT, index=False)
print(f'saved {PRED_OUT} ({len(pred_df)} patients)')

# ── 13. test-set error breakdown at each threshold ──
test_rows = pred_df[pred_df.split == 'test']
n_pos = int((test_rows.true_label == 1).sum())
print(f'\ntest set: {len(test_rows)} patients, {n_pos} real delirium cases')
for col, name in [('pred_label', 'argmax @ 0.5'),
                  ('pred_label_tuned', f'>={TARGET_RECALL:.0%} recall')]:
    tp = int(((test_rows[col] == 1) & (test_rows.true_label == 1)).sum())
    fp = int(((test_rows[col] == 1) & (test_rows.true_label == 0)).sum())
    fn = int(((test_rows[col] == 0) & (test_rows.true_label == 1)).sum())
    rec_t = tp / (tp + fn) if (tp + fn) else 0.0
    prec_t = tp / (tp + fp) if (tp + fp) else 0.0
    print(f'  {name:14s}: caught {tp}/{n_pos} (recall {rec_t:.2f}), '
          f'{fp} false alarms (precision {prec_t:.2f}), {fp + fn} total errors')

print('\nworst false negatives (true delirium ranked lowest, test set):')
print(test_rows[test_rows.true_label == 1].sort_values('prob_delirium').head(10).to_string(index=False))