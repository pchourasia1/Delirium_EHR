"""
Load the patient-feature heterogeneous graph into NetworkX and
print structural stats + visualize a sampled subgraph.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── load CSVs ──
patients = pd.read_csv('nodes_patient.csv')
fv_nodes = pd.read_csv('nodes_feature_value.csv')
edges = pd.read_csv('edges_patient_feature.csv')

# ── build full graph ──
G = nx.Graph()

for _, row in patients.iterrows():
    G.add_node(
        f"p_{row['patient_node_id']}",
        node_type='patient',
        label=int(row['delirium_label']),
        stay_id=int(row['stay_id']),
    )

for _, row in fv_nodes.iterrows():
    G.add_node(
        f"fv_{row['fv_node_id']}",
        node_type='feature_value',
        feature=row['feature_name'],
        level=row['level_label'],
    )

for _, row in edges.iterrows():
    G.add_edge(f"p_{row['patient_node_id']}", f"fv_{row['fv_node_id']}")

# ── print structure stats ──
p_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'patient']
f_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'feature_value']

print('=' * 50)
print('GRAPH STRUCTURE SUMMARY')
print('=' * 50)
print(f'Total nodes:          {G.number_of_nodes()}')
print(f'  patient nodes:      {len(p_nodes)}')
print(f'  feature_value nodes:{len(f_nodes)}')
print(f'Total edges:          {G.number_of_edges()}')
print(f'Graph density:        {nx.density(G):.6f}')
print(f'Is bipartite:         {nx.is_bipartite(G)}')
print()

# degree stats for each node type
p_deg = [G.degree(n) for n in p_nodes]
f_deg = [G.degree(n) for n in f_nodes]
print('Patient node degrees:')
print(f'  min={min(p_deg)}  max={max(p_deg)}  mean={np.mean(p_deg):.1f}')
print()
print('Feature_value node degrees (patients sharing that value):')
print(f'  min={min(f_deg)}  max={max(f_deg)}  mean={np.mean(f_deg):.1f}')
print()

# per-feature breakdown
print(f'{"feature_value node":<40s} {"feature":<30s} degree')
print('-' * 85)
for n in sorted(f_nodes):
    d = G.nodes[n]
    print(f'{n:<40s} {d["feature"]:<30s} {G.degree(n)}')

# ── visualize sampled subgraph ──
np.random.seed(42)
pos_ids = [n for n in p_nodes if G.nodes[n]['label'] == 1]
neg_ids = [n for n in p_nodes if G.nodes[n]['label'] == 0]
sample = list(np.random.choice(pos_ids, size=5, replace=False)) + \
         list(np.random.choice(neg_ids, size=5, replace=False))

# collect all feature_value neighbors of sampled patients
fv_neighbors = set()
for p in sample:
    fv_neighbors.update(G.neighbors(p))

sub = G.subgraph(sample + list(fv_neighbors)).copy()

# layout: patients on the left, feature_value nodes on the right
pos = {}
p_sub = [n for n in sub if sub.nodes[n]['node_type'] == 'patient']
f_sub = [n for n in sub if sub.nodes[n]['node_type'] == 'feature_value']

for i, n in enumerate(sorted(p_sub)):
    pos[n] = (0, -i)
for i, n in enumerate(sorted(f_sub)):
    pos[n] = (3, -i * (len(p_sub) / len(f_sub)))

# colors
node_colors = []
for n in sub.nodes():
    d = sub.nodes[n]
    if d['node_type'] == 'patient':
        node_colors.append('#e74c3c' if d['label'] == 1 else '#3498db')
    else:
        node_colors.append('#2ecc71')

node_sizes = [300 if sub.nodes[n]['node_type'] == 'patient' else 150 for n in sub.nodes()]

# labels: short labels for feature_value nodes
labels = {}
for n in sub.nodes():
    d = sub.nodes[n]
    if d['node_type'] == 'patient':
        labels[n] = f"{'DEL' if d['label']==1 else 'NEG'}"
    else:
        feat_short = d['feature'][:12]
        labels[n] = f"{feat_short}\n{d['level']}"

fig, ax = plt.subplots(figsize=(16, 12))
nx.draw(
    sub, pos, ax=ax,
    node_color=node_colors,
    node_size=node_sizes,
    edge_color='#bdc3c7',
    alpha=0.9,
    with_labels=True,
    labels=labels,
    font_size=5,
    width=0.5,
)

legend = [
    mpatches.Patch(color='#e74c3c', label='Patient (delirium+)'),
    mpatches.Patch(color='#3498db', label='Patient (delirium-)'),
    mpatches.Patch(color='#2ecc71', label='Feature value node'),
]
ax.legend(handles=legend, loc='upper left', fontsize=10)
ax.set_title('Sampled Subgraph: 5 delirium+ / 5 delirium- patients', fontsize=14)
plt.tight_layout()
plt.savefig('graph_structure.png', dpi=150)
plt.show()
print('\nSaved graph_structure.png')