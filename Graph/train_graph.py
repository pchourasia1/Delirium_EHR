"""
Heterogeneous GAT for ICU delirium prediction using PyTorch Geometric.

Graph structure:
    patient nodes      -> 23 raw Boruta feature values as initial features
    feature_value nodes -> learnable embeddings (trained end-to-end)
    edge types:
        (patient, has_value, feature_value)    - clinical profile edges
        (feature_value, rev_has_value, patient) - reverse for message passing
        (patient, similar_to, patient)         - distance-weighted KNN edges

Loss: focal loss with class weights
Split: 70% train, 15% val, 15% test (stratified)

Requires: torch, torch_geometric, pandas, numpy, scikit-learn
"""

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
)
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv

# ── configuration ──
SEED = 42
HIDDEN_DIM = 128
FV_EMBED_DIM = 64
LR = 0.001
WEIGHT_DECAY = 1e-3
EPOCHS = 200
DROPOUT = 0.5
GAT_HEADS = 4
FOCAL_GAMMA = 2.0

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── focal loss ──
class FocalLoss(nn.Module):
    """
    Focal loss downweights easy-to-classify samples and focuses
    learning on hard cases near the decision boundary.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha    # class weight tensor
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ── 1. load data ──
cohort = pd.read_csv('boruta_cohort.csv')
fv_nodes = pd.read_csv('nodes_feature_value.csv')
edges_pf = pd.read_csv('edges_patient_feature.csv')
edges_pp = pd.read_csv('edges_patient_patient.csv')

ID_COLS = ['stay_id', 'subject_id', 'hadm_id']
LABEL_COL = 'delirium_label'
feat_cols = [c for c in cohort.columns if c not in ID_COLS + [LABEL_COL]]

# ── 2. prepare node features ──
labels = torch.tensor(cohort[LABEL_COL].values, dtype=torch.long)
num_fv = len(fv_nodes)

# split computed BEFORE scaling so the scaler never sees val/test rows
indices = np.arange(len(labels))
train_idx, temp_idx = train_test_split(
    indices, test_size=0.30, stratify=labels.numpy(), random_state=SEED
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, stratify=labels.numpy()[temp_idx], random_state=SEED
)

# fit scaler on training rows only, then transform every patient
scaler = StandardScaler()
scaler.fit(cohort[feat_cols].values[train_idx])
patient_x = torch.tensor(
    scaler.transform(cohort[feat_cols].values), dtype=torch.float
)

# ── 3. build HeteroData ──
data = HeteroData()

data['patient'].x = patient_x
data['patient'].y = labels
data['feature_value'].num_nodes = num_fv

# patient <-> feature_value edges
pf_src = torch.tensor(edges_pf['patient_node_id'].values, dtype=torch.long)
pf_dst = torch.tensor(edges_pf['fv_node_id'].values, dtype=torch.long)
data['patient', 'has_value', 'feature_value'].edge_index = torch.stack([pf_src, pf_dst])
data['feature_value', 'rev_has_value', 'patient'].edge_index = torch.stack([pf_dst, pf_src])

# patient <-> patient KNN edges with distance weights
pp_src = torch.tensor(edges_pp['src_patient_id'].values, dtype=torch.long)
pp_dst = torch.tensor(edges_pp['dst_patient_id'].values, dtype=torch.long)
pp_weight = torch.tensor(edges_pp['weight'].values, dtype=torch.float).unsqueeze(-1)  # (E, 1)
data['patient', 'similar_to', 'patient'].edge_index = torch.stack([pp_src, pp_dst])
data['patient', 'similar_to', 'patient'].edge_attr = pp_weight

print(f'Patient-feature edges: {len(edges_pf)}')
print(f'Patient-patient edges: {len(edges_pp)}')
print(f'Edge weight range: [{pp_weight.min().item():.4f}, {pp_weight.max().item():.4f}]')

# ── 4. stratified train / val / test masks (70 / 15 / 15), split computed in section 2 ──
for name, idx in [('train_mask', train_idx), ('val_mask', val_idx), ('test_mask', test_idx)]:
    mask = torch.zeros(len(labels), dtype=torch.bool)
    mask[idx] = True
    data['patient'][name] = mask

print(f'Train: {train_idx.shape[0]}  Val: {val_idx.shape[0]}  Test: {test_idx.shape[0]}')
print(f'Train pos rate: {labels[train_idx].float().mean():.4f}')
print(f'Val   pos rate: {labels[val_idx].float().mean():.4f}')
print(f'Test  pos rate: {labels[test_idx].float().mean():.4f}')

# ── 5. class weights ──
num_pos = labels[train_idx].sum().item()
num_neg = len(train_idx) - num_pos
class_weight = torch.tensor([1.0, (num_neg / num_pos) ** 0.5], dtype=torch.float)
print(f'Class weights: [neg={class_weight[0]:.2f}, pos={class_weight[1]:.2f}]')


# ── 6. model (2 layers, distance-weighted KNN edges) ──
class HeteroGAT(nn.Module):
    def __init__(self, patient_in, fv_embed_dim, hidden, num_fv, dropout, heads):
        super().__init__()
        self.fv_embedding = nn.Embedding(num_fv, fv_embed_dim)
        self.patient_in = patient_in

        # layer 1
        self.conv1 = HeteroConv({
            ('patient', 'has_value', 'feature_value'):
                GATConv((patient_in, fv_embed_dim), hidden // heads, heads=heads, add_self_loops=False),
            ('feature_value', 'rev_has_value', 'patient'):
                GATConv((fv_embed_dim, patient_in), hidden // heads, heads=heads, add_self_loops=False),
            ('patient', 'similar_to', 'patient'):
                GATConv(patient_in, hidden // heads, heads=heads, add_self_loops=False, edge_dim=1),
        })
        self.norm1 = nn.LayerNorm(hidden)

        # layer 2
        self.conv2 = HeteroConv({
            ('patient', 'has_value', 'feature_value'):
                GATConv((hidden, hidden), hidden // heads, heads=heads, add_self_loops=False),
            ('feature_value', 'rev_has_value', 'patient'):
                GATConv((hidden, hidden), hidden // heads, heads=heads, add_self_loops=False),
            ('patient', 'similar_to', 'patient'):
                GATConv(hidden, hidden // heads, heads=heads, add_self_loops=False, edge_dim=1),
        })
        self.norm2 = nn.LayerNorm(hidden)

        # classifier: skip connection concatenates raw features with GNN output
        self.classifier = nn.Sequential(
            nn.Linear(hidden + patient_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.dropout = dropout

    def forward(self, data):
        x_raw = data['patient'].x  # save for skip connection

        x_dict = {
            'patient': data['patient'].x,
            'feature_value': self.fv_embedding.weight,
        }
        edge_index_dict = {
            ('patient', 'has_value', 'feature_value'):
                data['patient', 'has_value', 'feature_value'].edge_index,
            ('feature_value', 'rev_has_value', 'patient'):
                data['feature_value', 'rev_has_value', 'patient'].edge_index,
            ('patient', 'similar_to', 'patient'):
                data['patient', 'similar_to', 'patient'].edge_index,
        }
        edge_attr_dict = {
            ('patient', 'similar_to', 'patient'):
                data['patient', 'similar_to', 'patient'].edge_attr,
        }

        # layer 1
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        x_dict['patient'] = self.norm1(x_dict['patient'])
        x_dict = {k: F.relu(F.dropout(v, p=self.dropout, training=self.training))
                  for k, v in x_dict.items()}

        # layer 2 (residual on the patient stream preserves node identity)
        x_patient_res = x_dict['patient']
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        x_dict['patient'] = self.norm2(x_dict['patient']) + x_patient_res
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # skip connection: concat raw features with graph-learned features
        patient_out = torch.cat([x_dict['patient'], x_raw], dim=1)
        return self.classifier(patient_out)


# ── 7. training setup ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = HeteroGAT(
    patient_in=len(feat_cols),
    fv_embed_dim=FV_EMBED_DIM,
    hidden=HIDDEN_DIM,
    num_fv=num_fv,
    dropout=DROPOUT,
    heads=GAT_HEADS,
).to(device)

data = data.to(device)
class_weight = class_weight.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
criterion = FocalLoss(alpha=class_weight, gamma=FOCAL_GAMMA)


# ── 8. train / eval functions ──
def train_epoch():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data['patient'].train_mask], data['patient'].y[data['patient'].train_mask])
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

    y_true = data['patient'].y[mask].cpu().numpy()
    y_pred = preds[mask].cpu().numpy()
    y_prob = probs[mask].cpu().numpy()

    return {
        'accuracy':   accuracy_score(y_true, y_pred),
        'precision':  precision_score(y_true, y_pred, zero_division=0),
        'recall':     recall_score(y_true, y_pred, zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro':   f1_score(y_true, y_pred, average='macro'),
        'roc_auc':    roc_auc_score(y_true, y_prob),
        'y_true': y_true,
        'y_pred': y_pred,
    }


# ── 9. training loop (saves best checkpoint, no early stopping) ──
best_val_auc = 0.0
best_epoch = 0
best_state = None

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
            best_val_auc = val['roc_auc']
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = '  *best'

        print(f'{epoch:5d}  {loss:7.4f}  {lr_now:8.5f}  {trn["roc_auc"]:8.4f}  {val["f1_macro"]:7.4f}  {val["roc_auc"]:8.4f}{marker}')

print(f'\nBest val AUC: {best_val_auc:.4f} at epoch {best_epoch}')

# ── 10. test evaluation using best checkpoint ──
model.load_state_dict(best_state)
test = evaluate(data['patient'].test_mask)

print('\n' + '=' * 50)
print('TEST SET RESULTS (best checkpoint)')
print('=' * 50)
print(f'Accuracy:    {test["accuracy"]:.4f}')
print(f'Precision:   {test["precision"]:.4f}')
print(f'Recall:      {test["recall"]:.4f}')
print(f'F1 weighted: {test["f1_weighted"]:.4f}')
print(f'F1 macro:    {test["f1_macro"]:.4f}')
print(f'ROC AUC:     {test["roc_auc"]:.4f}')
print()
print(classification_report(
    test['y_true'], test['y_pred'],
    target_names=['No Delirium', 'Delirium'],
))