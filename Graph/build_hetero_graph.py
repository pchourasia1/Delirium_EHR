"""
Build a patient-feature heterogeneous graph from boruta_cohort.csv.

Node types:
    patient        - one node per row (stay)
    feature_value  - one node per distinct (feature, level/bin) pair

Edge types:
    (patient, has_value, feature_value)  - one per feature per patient
    (patient, similar_to, patient)       - KNN edges with distance weights

Outputs:
    nodes_patient.csv           patient_node_id, stay_id, subject_id, hadm_id, delirium_label
    nodes_feature_value.csv     fv_node_id, feature_name, level_label
    edges_patient_feature.csv   patient_node_id, fv_node_id
    edges_patient_patient.csv   src_patient_id, dst_patient_id, weight
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

INPUT_FILE = 'boruta_cohort.csv'
ID_COLS = ['stay_id', 'subject_id', 'hadm_id']
LABEL_COL = 'delirium_label'
LEVEL_THRESHOLD = 10
N_BINS = 10
K_NEIGHBORS = 10


def main():
    df = pd.read_csv(INPUT_FILE)
    features = [c for c in df.columns if c not in ID_COLS + [LABEL_COL]]

    # ── build feature_value nodes ──
    fv_lookup = {}
    fv_table = []
    patient_level = {}
    next_id = 0

    for feat in features:
        if df[feat].nunique() <= LEVEL_THRESHOLD:
            level_labels = df[feat].astype(str)
        else:
            level_labels = pd.qcut(df[feat], q=N_BINS, duplicates='drop').astype(str)

        patient_level[feat] = level_labels

        for lvl in sorted(level_labels.unique()):
            key = (feat, lvl)
            if key not in fv_lookup:
                fv_lookup[key] = next_id
                fv_table.append((next_id, feat, lvl))
                next_id += 1

    # ── build patient-feature edges ──
    pf_src, pf_dst = [], []
    for feat in features:
        fv_ids = patient_level[feat].map(lambda lvl, f=feat: fv_lookup[(f, lvl)]).to_numpy()
        pf_src.append(np.arange(len(df)))
        pf_dst.append(fv_ids)
    pf_src = np.concatenate(pf_src)
    pf_dst = np.concatenate(pf_dst)

    # ── build patient-patient KNN edges with distance weights ──
    print(f'Computing {K_NEIGHBORS}-NN on {len(df)} patients...')
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].values)

    nn_model = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric='euclidean', n_jobs=-1)
    nn_model.fit(X)
    distances, indices = nn_model.kneighbors(X)

    # skip self (column 0), convert distance to weight
    pp_src, pp_dst, pp_weight = [], [], []
    for i in range(len(df)):
        neighbors = indices[i, 1:]
        dists = distances[i, 1:]
        weights = 1.0 / (1.0 + dists)  # closer = higher weight

        pp_src.append(np.full(K_NEIGHBORS, i))
        pp_dst.append(neighbors)
        pp_weight.append(weights)

    pp_src = np.concatenate(pp_src)
    pp_dst = np.concatenate(pp_dst)
    pp_weight = np.concatenate(pp_weight)

    # ── write outputs ──
    patients_out = pd.DataFrame({
        'patient_node_id': np.arange(len(df)),
        'stay_id': df['stay_id'].values,
        'subject_id': df['subject_id'].values,
        'hadm_id': df['hadm_id'].values,
        'delirium_label': df[LABEL_COL].values,
    })
    patients_out.to_csv('nodes_patient.csv', index=False)

    fv_out = pd.DataFrame(fv_table, columns=['fv_node_id', 'feature_name', 'level_label'])
    fv_out.to_csv('nodes_feature_value.csv', index=False)

    edges_pf = pd.DataFrame({'patient_node_id': pf_src, 'fv_node_id': pf_dst})
    edges_pf.to_csv('edges_patient_feature.csv', index=False)

    edges_pp = pd.DataFrame({'src_patient_id': pp_src, 'dst_patient_id': pp_dst, 'weight': pp_weight})
    edges_pp.to_csv('edges_patient_patient.csv', index=False)

    print(f'patient nodes:            {len(patients_out)}')
    print(f'feature_value nodes:      {len(fv_out)}')
    print(f'patient-feature edges:    {len(edges_pf)}')
    print(f'patient-patient edges:    {len(edges_pp)}')
    print(f'pf edges per patient:     {len(edges_pf) / len(patients_out):.1f}')
    print(f'pp edges per patient:     {K_NEIGHBORS}')
    print(f'edge weight range:        [{pp_weight.min():.4f}, {pp_weight.max():.4f}]')
    print(f'edge weight mean:         {pp_weight.mean():.4f}')
    print(f'positive rate:            {patients_out["delirium_label"].mean():.4f}')


if __name__ == '__main__':
    main()