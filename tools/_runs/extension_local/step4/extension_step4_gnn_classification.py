# Auto-generated from: extension_step4_gnn_classification.ipynb
from __future__ import annotations

IN_COLAB = False
DRIVE_BASE = ""

# Avoid Windows console UnicodeEncodeError (cp1252) for symbols/emoji printed in notebooks.
import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[attr-defined]
except Exception:
    pass

# --- skipped cell 0 (colab-only) ---

# --- cell 1 ---
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- cell 2 ---
# Import PyTorch Geometric
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
    from torch_geometric.loader import DataLoader as PyGDataLoader
    PYG_AVAILABLE = True
    print("PyTorch Geometric loaded successfully")
except ImportError:
    PYG_AVAILABLE = False
    print("PyTorch Geometric not available. Using simple baseline only.")

# --- cell 4 ---
# Load realized graphs from Substep 3
DATA_PATH = "extension_data/realized_task_graphs.pkl"

if IN_COLAB:
    DRIVE_DATA_PATH = "/content/drive/MyDrive/AML_Project/extension_data/realized_task_graphs.pkl"
    if os.path.exists(DRIVE_DATA_PATH):
        DATA_PATH = DRIVE_DATA_PATH

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, 'rb') as f:
        loaded_data = pickle.load(f)

    realized_graphs = loaded_data['realized_graphs']
    splits = loaded_data['splits']
    config = loaded_data['config']

    print(f"Loaded {len(realized_graphs)} realized graphs")
    print(f"Feature dimension: {config['feature_dim']}")
else:
    print(f"Data not found at {DATA_PATH}")
    print("Please run extension_step3_task_graph_matching.ipynb first!")

# --- cell 5 ---
FEATURE_DIM = config['feature_dim']

# Group by recipe type
recipe_groups = defaultdict(list)
for recording_id, graph in realized_graphs.items():
    recipe_groups[graph['recipe_id']].append(recording_id)

print(f"Number of recipe types: {len(recipe_groups)}")

# Statistics
num_nodes_list = [g['num_nodes'] for g in realized_graphs.values()]
labels = [g['recipe_label'] for g in realized_graphs.values()]
print(f"Average nodes per graph: {np.mean(num_nodes_list):.1f}")
print(f"Label distribution: {sum(labels)} incorrect, {len(labels)-sum(labels)} correct")

# --- cell 7 ---
def create_pyg_data(realized_graph, recording_id):
    """
    Convert realized graph to PyTorch Geometric Data object.
    """
    # Node features
    x = torch.FloatTensor(realized_graph['node_features'])

    # Edge index (make undirected by adding reverse edges)
    edge_index = realized_graph['edge_index']
    if edge_index.shape[1] > 0:
        # Add reverse edges for undirected graph
        reverse_edges = np.array([edge_index[1], edge_index[0]])
        edge_index = np.concatenate([edge_index, reverse_edges], axis=1)
    edge_index = torch.LongTensor(edge_index)

    # Label
    y = torch.FloatTensor([realized_graph['recipe_label']])

    # Additional features
    node_matched = torch.FloatTensor(realized_graph['node_matched'].astype(float))
    match_sims = torch.FloatTensor(realized_graph['match_similarities'])

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        node_matched=node_matched,
        match_sims=match_sims,
        recording_id=recording_id,
        recipe_id=realized_graph['recipe_id']
    )

    return data

if PYG_AVAILABLE:
    # Create PyG dataset
    pyg_data = {}
    for recording_id, graph in realized_graphs.items():
        pyg_data[recording_id] = create_pyg_data(graph, recording_id)

    print(f"Created {len(pyg_data)} PyG Data objects")

    # Sample
    sample = list(pyg_data.values())[0]
    print(f"\nSample graph:")
    print(f"  Nodes: {sample.x.shape}")
    print(f"  Edges: {sample.edge_index.shape}")
    print(f"  Label: {sample.y.item()}")

# --- cell 9 ---
if PYG_AVAILABLE:

    class GCNClassifier(nn.Module):
        """Graph Convolutional Network for graph classification."""

        def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
            super().__init__()

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()

            # First layer
            self.convs.append(GCNConv(input_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

            # Hidden layers
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))

            self.dropout = nn.Dropout(dropout)

            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x, edge_index, batch):
            # GCN layers
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = self.dropout(x)

            # Graph-level pooling (mean + max)
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=-1)

            return self.classifier(x)


    class GATClassifier(nn.Module):
        """Graph Attention Network for graph classification."""

        def __init__(self, input_dim, hidden_dim=128, num_layers=3, heads=4, dropout=0.3):
            super().__init__()

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()

            # First layer
            self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

            # Hidden layers
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
                self.bns.append(nn.BatchNorm1d(hidden_dim))

            self.dropout = nn.Dropout(dropout)

            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x, edge_index, batch):
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.elu(x)
                x = self.dropout(x)

            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=-1)

            return self.classifier(x)


    class GraphSAGEClassifier(nn.Module):
        """GraphSAGE for graph classification."""

        def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
            super().__init__()

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()

            self.convs.append(SAGEConv(input_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))

            self.dropout = nn.Dropout(dropout)

            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x, edge_index, batch):
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = self.dropout(x)

            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=-1)

            return self.classifier(x)

    print("GNN models defined successfully")

# --- cell 10 ---
# Simple baseline without GNN (for comparison)
class SimplePoolingClassifier(nn.Module):
    """Simple baseline: Pool node features and classify."""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, batch):
        # Simple mean + max pooling
        if PYG_AVAILABLE:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
        else:
            # Manual pooling
            x_mean = x.mean(dim=0, keepdim=True)
            x_max = x.max(dim=0, keepdim=True)[0]

        x = torch.cat([x_mean, x_max], dim=-1)
        return self.fc(x)

print("Simple baseline defined")

# --- cell 12 ---
def train_epoch_gnn(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.view(-1, 1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

        preds = torch.sigmoid(out).detach().cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(batch.y.cpu().numpy().flatten())

    return total_loss / len(loader.dataset), np.array(all_preds), np.array(all_labels)


def evaluate_gnn(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.view(-1, 1))

            total_loss += loss.item() * batch.num_graphs

            preds = torch.sigmoid(out).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch.y.cpu().numpy().flatten())

    return total_loss / len(loader.dataset), np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels, threshold=0.5):
    binary_preds = (preds >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(labels, binary_preds),
        'precision': precision_score(labels, binary_preds, zero_division=0),
        'recall': recall_score(labels, binary_preds, zero_division=0),
        'f1': f1_score(labels, binary_preds, zero_division=0),
    }

    try:
        metrics['auc'] = roc_auc_score(labels, preds)
    except:
        metrics['auc'] = 0.5

    return metrics

# --- cell 14 ---
def leave_one_recipe_out_gnn(model_class, pyg_data, recipe_groups,
                              input_dim, num_epochs=30, lr=1e-3,
                              batch_size=32, pos_weight=2.0, **model_kwargs):
    """
    Leave-one-recipe-out cross-validation for GNN models.
    """
    recipe_ids = list(recipe_groups.keys())
    all_results = []

    print(f"\nRunning Leave-One-Recipe-Out evaluation with {len(recipe_ids)} folds")

    for fold_idx, test_recipe in enumerate(tqdm(recipe_ids, desc="Folds")):
        # Split data
        train_ids = []
        test_ids = recipe_groups[test_recipe]

        for recipe, recordings in recipe_groups.items():
            if recipe != test_recipe:
                train_ids.extend(recordings)

        # Filter to available data
        train_ids = [rid for rid in train_ids if rid in pyg_data]
        test_ids = [rid for rid in test_ids if rid in pyg_data]

        if len(test_ids) < 2 or len(train_ids) < 10:
            continue

        # Create data lists
        train_data = [pyg_data[rid] for rid in train_ids]
        test_data = [pyg_data[rid] for rid in test_ids]

        train_loader = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = PyGDataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Initialize model
        model = model_class(input_dim, **model_kwargs).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Training
        best_loss = float('inf')
        best_model_state = None

        for epoch in range(num_epochs):
            train_loss, _, _ = train_epoch_gnn(model, train_loader, optimizer, criterion, device)
            scheduler.step()

            if train_loss < best_loss:
                best_loss = train_loss
                best_model_state = model.state_dict().copy()

        if best_model_state:
            model.load_state_dict(best_model_state)

        # Evaluate
        test_loss, test_preds, test_labels = evaluate_gnn(model, test_loader, criterion, device)
        metrics = compute_metrics(test_preds, test_labels)

        all_results.append({
            'fold': fold_idx,
            'test_recipe': test_recipe,
            'train_size': len(train_ids),
            'test_size': len(test_ids),
            **metrics
        })

    return all_results


def summarize_results(results, model_name):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    print(f"\n{'='*60}")
    print(f"{model_name} Results (Leave-One-Recipe-Out)")
    print(f"{'='*60}")

    for metric in metrics:
        values = [r[metric] for r in results]
        print(f"{metric.capitalize():12s}: {np.mean(values)*100:.2f}% ± {np.std(values)*100:.2f}%")

    print(f"{'='*60}")

    return {
        metric: {'mean': np.mean([r[metric] for r in results]),
                 'std': np.std([r[metric] for r in results])}
        for metric in metrics
    }

# --- cell 16 ---
# Hyperparameters
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
POS_WEIGHT = 2.0
HIDDEN_DIM = 128

all_model_results = {}

# --- cell 17 ---
if PYG_AVAILABLE:
    # Simple Pooling Baseline
    print("\n" + "="*60)
    print("Training Simple Pooling Baseline")
    print("="*60)

    pooling_results = leave_one_recipe_out_gnn(
        SimplePoolingClassifier,
        pyg_data,
        recipe_groups,
        input_dim=FEATURE_DIM,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        pos_weight=POS_WEIGHT,
        hidden_dim=HIDDEN_DIM
    )

    all_model_results['SimplePooling'] = summarize_results(pooling_results, "Simple Pooling Baseline")

# --- cell 18 ---
if PYG_AVAILABLE:
    # GCN
    print("\n" + "="*60)
    print("Training GCN")
    print("="*60)

    gcn_results = leave_one_recipe_out_gnn(
        GCNClassifier,
        pyg_data,
        recipe_groups,
        input_dim=FEATURE_DIM,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        pos_weight=POS_WEIGHT,
        hidden_dim=HIDDEN_DIM,
        num_layers=3
    )

    all_model_results['GCN'] = summarize_results(gcn_results, "GCN")

# --- cell 19 ---
if PYG_AVAILABLE:
    # GAT
    print("\n" + "="*60)
    print("Training GAT")
    print("="*60)

    gat_results = leave_one_recipe_out_gnn(
        GATClassifier,
        pyg_data,
        recipe_groups,
        input_dim=FEATURE_DIM,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        pos_weight=POS_WEIGHT,
        hidden_dim=HIDDEN_DIM,
        num_layers=3,
        heads=4
    )

    all_model_results['GAT'] = summarize_results(gat_results, "GAT")

# --- cell 20 ---
if PYG_AVAILABLE:
    # GraphSAGE
    print("\n" + "="*60)
    print("Training GraphSAGE")
    print("="*60)

    sage_results = leave_one_recipe_out_gnn(
        GraphSAGEClassifier,
        pyg_data,
        recipe_groups,
        input_dim=FEATURE_DIM,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        pos_weight=POS_WEIGHT,
        hidden_dim=HIDDEN_DIM,
        num_layers=3
    )

    all_model_results['GraphSAGE'] = summarize_results(sage_results, "GraphSAGE")

# --- cell 22 ---
import pandas as pd
import matplotlib.pyplot as plt

# Create comparison table
comparison_data = []
for model_name, results in all_model_results.items():
    row = {'Model': model_name}
    for metric, values in results.items():
        row[f'{metric.capitalize()}'] = f"{values['mean']*100:.1f}%"
    comparison_data.append(row)

df = pd.DataFrame(comparison_data)
print("\n" + "="*60)
print("GNN Model Comparison (Leave-One-Recipe-Out)")
print("="*60)
print(df.to_string(index=False))

# --- cell 23 ---
# Visualization
if len(all_model_results) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics))
    width = 0.2

    colors = ['#95a5a6', '#2ecc71', '#3498db', '#e74c3c']
    for i, (model_name, results) in enumerate(all_model_results.items()):
        means = [results[m]['mean'] * 100 for m in metrics]
        stds = [results[m]['std'] * 100 for m in metrics]
        axes[0].bar(x + i*width, means, width, yerr=stds, label=model_name,
                    color=colors[i % len(colors)], alpha=0.8, capsize=3)

    axes[0].set_xlabel('Metric')
    axes[0].set_ylabel('Score (%)')
    axes[0].set_title('GNN Models Comparison')
    axes[0].set_xticks(x + width * (len(all_model_results) - 1) / 2)
    axes[0].set_xticklabels([m.capitalize() for m in metrics])
    axes[0].legend()
    axes[0].set_ylim(0, 100)

    # F1 across models
    model_names = list(all_model_results.keys())
    f1_means = [all_model_results[m]['f1']['mean'] * 100 for m in model_names]
    f1_stds = [all_model_results[m]['f1']['std'] * 100 for m in model_names]

    bars = axes[1].bar(model_names, f1_means, yerr=f1_stds, capsize=5,
                       color=colors[:len(model_names)], alpha=0.8)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('F1 Score (%)')
    axes[1].set_title('F1 Score Comparison')
    axes[1].set_ylim(0, 100)

    for bar, mean in zip(bars, f1_means):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f'{mean:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('extension_data/gnn_comparison.png', dpi=150)
    plt.show()

# --- cell 24 ---
# Save all results
import json

def convert_numpy(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj

results_to_save = {
    'summary': all_model_results,
    'config': {
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'pos_weight': POS_WEIGHT,
        'hidden_dim': HIDDEN_DIM,
        'feature_dim': FEATURE_DIM
    }
}

if PYG_AVAILABLE:
    results_to_save['gcn_folds'] = gcn_results if 'gcn_results' in dir() else []
    results_to_save['gat_folds'] = gat_results if 'gat_results' in dir() else []
    results_to_save['sage_folds'] = sage_results if 'sage_results' in dir() else []

with open('extension_data/gnn_results.json', 'w') as f:
    json.dump(convert_numpy(results_to_save), f, indent=2)

print("Results saved to extension_data/gnn_results.json")

if __name__ == "__main__":
    pass
