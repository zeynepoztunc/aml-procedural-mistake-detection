# Auto-generated from a Jupyter notebook.
# Source: extension_step3_task_graph_matching.ipynb
#
# Notes:
# - Lines starting with !/%/%% are commented out (IPython-only).
# - Run from the repo root (folder containing requirements.txt).

# %% [cell 0]
# (colab-only setup cell omitted)

# %% [cell 1]
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [cell 2]
# ## 1. Load Data and Task Graphs

# %% [cell 3]
# Load processed step embeddings from Substep 1
DATA_PATH = "extension_data/step_embeddings_gt.pkl"

if IN_COLAB:
    DRIVE_DATA_PATH = "/content/drive/MyDrive/AML_Project/extension_data/step_embeddings_gt.pkl"
    if os.path.exists(DRIVE_DATA_PATH):
        DATA_PATH = DRIVE_DATA_PATH

with open(DATA_PATH, 'rb') as f:
    loaded_data = pickle.load(f)

processed_data = loaded_data['data']
splits = loaded_data['splits']
print(f"Loaded {len(processed_data)} recordings")

# Load step annotations for textual descriptions
with open('annotations/annotation_json/step_annotations.json', 'r') as f:
    step_annotations = json.load(f)

# %% [cell 4]
# Task graph annotations (from CaptainCook4D)
# Each recipe has a predefined task graph with step descriptions

TASK_GRAPH_PATH = "annotations/annotation_json/task_graphs.json"

# If task_graphs.json doesn't exist, we'll build it from step annotations
if os.path.exists(TASK_GRAPH_PATH):
    with open(TASK_GRAPH_PATH, 'r') as f:
        task_graphs = json.load(f)
    print(f"Loaded {len(task_graphs)} task graphs")
else:
    print("Task graphs file not found. Building from step annotations...")
    task_graphs = None

# %% [cell 5]
# Build task graph templates from annotations
# A task graph defines the standard sequence of steps for each recipe type

def build_task_graphs_from_annotations(step_annotations):
    """
    Build task graphs from step annotations.
    Each recipe type (activity_id) has a canonical set of steps.

    Note: activity_id is extracted from recording_id prefix (format: X_Y where X is activity/recipe number)
    """
    recipe_steps = defaultdict(dict)

    for recording_id, ann in step_annotations.items():
        # Extract recipe_id from recording_id prefix (format: "X_Y" where X is activity number)
        recipe_id = recording_id.split('_')[0] if '_' in recording_id else 'unknown'

        for step in ann.get('steps', []):
            step_id = step.get('step_id')
            description = step.get('description', '')

            if step_id and description:
                # Keep the most common/complete description
                if step_id not in recipe_steps[recipe_id]:
                    recipe_steps[recipe_id][step_id] = description

    # Convert to task graph format
    task_graphs = {}
    for recipe_id, steps in recipe_steps.items():
        sorted_steps = sorted(steps.items(), key=lambda x: x[0])

        task_graphs[recipe_id] = {
            'recipe_id': recipe_id,
            'nodes': [
                {'step_id': sid, 'description': desc, 'index': i}
                for i, (sid, desc) in enumerate(sorted_steps)
            ],
            'edges': [
                {'from': i, 'to': i+1}
                for i in range(len(sorted_steps) - 1)
            ]
        }

    return task_graphs

if task_graphs is None:
    task_graphs = build_task_graphs_from_annotations(step_annotations)

print(f"Task graphs for {len(task_graphs)} recipe types")
for recipe_id, graph in list(task_graphs.items())[:3]:
    print(f"  Recipe {recipe_id}: {len(graph['nodes'])} steps")

# %% [cell 6]
# ## 2. EgoVLP Text Encoder (Aligned with Visual Features)
#
# We load the EgoVLP model and use its text encoder.

# %% [cell 7]
# (colab-only setup cell omitted)

# %% [cell 8]
# ## 3. Encode Task Graph Nodes

# %% [cell 9]
def encode_task_graph_nodes(task_graph):
    """
    Encode all nodes in a task graph using the EgoVLP text encoder.

    Uses the encode_texts function which leverages EgoVLP for aligned
    text-visual embeddings (or falls back to sentence-transformers).

    Returns:
        node_embeddings: numpy array of shape (num_nodes, feature_dim)
        node_info: list of node metadata
    """
    nodes = task_graph['nodes']
    descriptions = [node['description'] for node in nodes]

    # Use the unified encode_texts function (EgoVLP or fallback)
    node_embeddings = encode_texts(descriptions, normalize=True)

    return node_embeddings.cpu().numpy(), nodes

# Encode all task graphs
print("Encoding task graph nodes...")
task_graph_embeddings = {}

for recipe_id, graph in tqdm(task_graphs.items(), desc="Encoding graphs"):
    embeddings, nodes = encode_task_graph_nodes(graph)
    task_graph_embeddings[recipe_id] = {
        'embeddings': embeddings,
        'nodes': nodes,
        'edges': graph['edges']
    }

print(f"Encoded {len(task_graph_embeddings)} task graphs")
print(f"Using {'EgoVLP (aligned)' if EGOVLP_AVAILABLE else 'Sentence-BERT (approximate)'} text encoder")

# %% [cell 10]
# ## 4. Hungarian Matching
#
# Match visual step embeddings to task graph nodes using the Hungarian algorithm.
# This ensures one-to-one matching between observed steps and graph nodes.

# %% [cell 11]
def compute_similarity_matrix(visual_embeddings, text_embeddings):
    """
    Compute cosine similarity matrix between visual and text embeddings.

    Args:
        visual_embeddings: (num_visual, feature_dim)
        text_embeddings: (num_text, feature_dim)

    Returns:
        similarity: (num_visual, num_text)
    """
    # Normalize
    visual_norm = visual_embeddings / (np.linalg.norm(visual_embeddings, axis=1, keepdims=True) + 1e-8)
    text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity
    similarity = np.dot(visual_norm, text_norm.T)

    return similarity


def hungarian_matching(visual_embeddings, text_embeddings):
    """
    Perform Hungarian matching between visual steps and task graph nodes.

    Args:
        visual_embeddings: (num_visual, feature_dim)
        text_embeddings: (num_text, feature_dim)

    Returns:
        matches: list of (visual_idx, text_idx, similarity) tuples
        unmatched_visual: indices of unmatched visual steps
        unmatched_text: indices of unmatched text nodes
    """
    num_visual = len(visual_embeddings)
    num_text = len(text_embeddings)

    # Compute similarity (higher is better)
    similarity = compute_similarity_matrix(visual_embeddings, text_embeddings)

    # Convert to cost (lower is better) for Hungarian algorithm
    cost = -similarity

    # Pad if sizes don't match
    if num_visual != num_text:
        max_size = max(num_visual, num_text)
        padded_cost = np.full((max_size, max_size), 0.0)  # Neutral cost for padding
        padded_cost[:num_visual, :num_text] = cost
        cost = padded_cost

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost)

    # Extract valid matches
    matches = []
    matched_visual = set()
    matched_text = set()

    for v_idx, t_idx in zip(row_ind, col_ind):
        if v_idx < num_visual and t_idx < num_text:
            sim = similarity[v_idx, t_idx]
            matches.append((v_idx, t_idx, sim))
            matched_visual.add(v_idx)
            matched_text.add(t_idx)

    unmatched_visual = [i for i in range(num_visual) if i not in matched_visual]
    unmatched_text = [i for i in range(num_text) if i not in matched_text]

    return matches, unmatched_visual, unmatched_text

# %% [cell 12]
# Test Hungarian matching on a sample
sample_recording = list(processed_data.keys())[0]
sample_data = processed_data[sample_recording]
# Convert to string to match dictionary keys
sample_recipe = str(sample_data['recipe_id'])

if sample_recipe in task_graph_embeddings:
    visual_emb = sample_data['step_embeddings']
    text_emb = task_graph_embeddings[sample_recipe]['embeddings']
    nodes = task_graph_embeddings[sample_recipe]['nodes']

    matches, unmatched_v, unmatched_t = hungarian_matching(visual_emb, text_emb)

    print(f"Recording: {sample_recording}")
    print(f"Recipe: {sample_recipe}")
    print(f"Visual steps: {len(visual_emb)}, Graph nodes: {len(text_emb)}")
    print(f"Matches: {len(matches)}")
    print(f"\nTop matches:")
    for v_idx, t_idx, sim in sorted(matches, key=lambda x: -x[2])[:5]:
        # Handle missing step_info by checking or falling back to annotations
        visual_desc = "N/A"
        if 'step_info' in sample_data and v_idx < len(sample_data['step_info']):
             visual_desc = sample_data['step_info'][v_idx].get('description', 'N/A')
        elif sample_recording in step_annotations:
             try:
                 visual_desc = step_annotations[sample_recording]['steps'][v_idx]['description']
             except (IndexError, KeyError):
                 visual_desc = f"Step {v_idx}"
        else:
             visual_desc = f"Step {v_idx}"

        visual_desc = str(visual_desc)[:50]
        text_desc = nodes[t_idx]['description'][:50]
        print(f"  Visual: '{visual_desc}...' -> Graph: '{text_desc}...' (sim={sim:.3f})")
else:
    print(f"Recipe ID {sample_recipe} (type: {type(sample_recipe)}) not found in task_graphs keys: {list(task_graph_embeddings.keys())[:5]}")

# %% [cell 13]
# ## 5. Create Realized Task Graphs
#
# For each recording, create a "realized" task graph by:
# 1. Matching visual steps to graph nodes
# 2. Updating node features with learned projection of (text + visual)

# %% [cell 14]
class FeatureProjector(nn.Module):
    """Learnable projection to combine visual and textual features."""

    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, visual_features, text_features):
        """
        Project combined visual and text features.

        Args:
            visual_features: (batch, feature_dim) or None for unmatched nodes
            text_features: (batch, feature_dim)
        """
        if visual_features is None:
            # For unmatched nodes, use zeros for visual
            visual_features = torch.zeros_like(text_features)

        combined = torch.cat([visual_features, text_features], dim=-1)
        return self.proj(combined)

# %% [cell 15]
def create_realized_graph(recording_data, task_graph_emb, feature_projector=None):
    """
    Create a realized task graph by matching visual steps to nodes.

    Returns:
        realized_graph: dict with
            - node_features: (num_nodes, feature_dim)
            - edge_index: (2, num_edges) - COO format for PyG
            - node_matched: boolean mask for matched nodes
            - match_similarities: matching scores
    """
    visual_emb = recording_data['step_embeddings']
    text_emb = task_graph_emb['embeddings']
    nodes = task_graph_emb['nodes']
    edges = task_graph_emb['edges']

    num_nodes = len(nodes)
    feature_dim = visual_emb.shape[1]

    # Perform matching
    matches, unmatched_v, unmatched_t = hungarian_matching(visual_emb, text_emb)

    # Initialize node features with text embeddings
    node_features = np.copy(text_emb)
    node_matched = np.zeros(num_nodes, dtype=bool)
    match_similarities = np.zeros(num_nodes)

    # Update matched nodes with combined features
    for v_idx, t_idx, sim in matches:
        node_matched[t_idx] = True
        match_similarities[t_idx] = sim

        if feature_projector is not None:
            # Use learned projection
            v_tensor = torch.FloatTensor(visual_emb[v_idx:v_idx+1])
            t_tensor = torch.FloatTensor(text_emb[t_idx:t_idx+1])
            with torch.no_grad():
                projected = feature_projector(v_tensor, t_tensor)
            node_features[t_idx] = projected.numpy().squeeze()
        else:
            # Simple: average visual and text
            node_features[t_idx] = 0.5 * visual_emb[v_idx] + 0.5 * text_emb[t_idx]

    # Convert edges to COO format
    if len(edges) > 0:
        edge_src = [e['from'] for e in edges]
        edge_dst = [e['to'] for e in edges]
        edge_index = np.array([edge_src, edge_dst])
    else:
        edge_index = np.zeros((2, 0), dtype=int)

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'node_matched': node_matched,
        'match_similarities': match_similarities,
        'num_matches': len(matches),
        'num_nodes': num_nodes,
        'num_visual_steps': len(visual_emb)
    }

# %% [cell 16]
# Process all recordings to create realized graphs
print("Creating realized task graphs for all recordings...")

realized_graphs = {}
missing_graphs = []

for recording_id, data in tqdm(processed_data.items(), desc="Processing"):
    # Convert to string to match dictionary keys
    recipe_id = str(data['recipe_id'])

    if recipe_id not in task_graph_embeddings:
        missing_graphs.append(recording_id)
        continue

    task_graph_emb = task_graph_embeddings[recipe_id]

    realized = create_realized_graph(data, task_graph_emb)
    realized['recipe_label'] = data['recipe_label']
    realized['recipe_id'] = recipe_id

    realized_graphs[recording_id] = realized

print(f"\nCreated {len(realized_graphs)} realized graphs")
print(f"Missing task graphs: {len(missing_graphs)}")

# Statistics
if len(realized_graphs) > 0:
    match_ratios = [g['num_matches'] / g['num_nodes'] for g in realized_graphs.values()]
    print(f"Average match ratio: {np.mean(match_ratios):.2%}")
else:
    print("No realized graphs created. Check dictionary keys.")

# %% [cell 17]
# ## 6. Analyze Matching Quality

# %% [cell 18]
import matplotlib.pyplot as plt

# Matching statistics
all_similarities = []
match_ratios = []
correct_sims = []
incorrect_sims = []

for g in realized_graphs.values():
    sims = g['match_similarities'][g['node_matched']]
    all_similarities.extend(sims)
    match_ratios.append(g['num_matches'] / g['num_nodes'])

    if g['recipe_label'] == 0:
        correct_sims.extend(sims)
    else:
        incorrect_sims.extend(sims)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Match similarity distribution
axes[0].hist(all_similarities, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Cosine Similarity')
axes[0].set_ylabel('Count')
axes[0].set_title('Match Similarity Distribution')
axes[0].axvline(np.mean(all_similarities), color='r', linestyle='--',
                label=f'Mean: {np.mean(all_similarities):.3f}')
axes[0].legend()

# Match ratio distribution
axes[1].hist(match_ratios, bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1].set_xlabel('Match Ratio (matches/nodes)')
axes[1].set_ylabel('Count')
axes[1].set_title('Match Ratio Distribution')

# Similarity by recipe correctness
axes[2].hist(correct_sims, bins=50, alpha=0.6, label='Correct recipes', color='green')
axes[2].hist(incorrect_sims, bins=50, alpha=0.6, label='Incorrect recipes', color='red')
axes[2].set_xlabel('Cosine Similarity')
axes[2].set_ylabel('Count')
axes[2].set_title('Similarity by Recipe Correctness')
axes[2].legend()

plt.tight_layout()
plt.savefig('extension_data/matching_analysis.png', dpi=150)
plt.show()

print(f"\nMatching Statistics:")
print(f"  Average similarity: {np.mean(all_similarities):.3f}")
print(f"  Average match ratio: {np.mean(match_ratios):.2%}")
print(f"  Correct recipes avg sim: {np.mean(correct_sims):.3f}")
print(f"  Incorrect recipes avg sim: {np.mean(incorrect_sims):.3f}")

# %% [cell 19]
# ## 7. Save Realized Graphs

# %% [cell 20]
# Save realized graphs for Substep 4
OUTPUT_DIR = "extension_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Determine text encoder type for config
text_encoder_type = 'egovlp' if EGOVLP_AVAILABLE else 'sentence-transformers-fallback'

save_data = {
    'realized_graphs': realized_graphs,
    'task_graph_embeddings': task_graph_embeddings,
    'splits': splits,
    'config': {
        'feature_dim': FEATURE_DIM,
        'text_encoder': text_encoder_type,
        'text_encoder_aligned': EGOVLP_AVAILABLE,  # True if using aligned EgoVLP embeddings
        'matching': 'hungarian'
    }
}

output_path = os.path.join(OUTPUT_DIR, "realized_task_graphs.pkl")
with open(output_path, 'wb') as f:
    pickle.dump(save_data, f)

print(f"Saved realized graphs to {output_path}")
print(f"Text encoder: {text_encoder_type} (aligned: {EGOVLP_AVAILABLE})")

# Also save to Drive if in Colab
if IN_COLAB:
    drive_output = os.path.join("/content/drive/MyDrive/AML_Project", "extension_data")
    os.makedirs(drive_output, exist_ok=True)
    drive_path = os.path.join(drive_output, "realized_task_graphs.pkl")
    with open(drive_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Also saved to Drive: {drive_path}")

# %% [cell 21]
# ## 8. Visualize Sample Graph

# %% [cell 22]
try:
    import networkx as nx

    # Visualize a sample realized graph
    sample_id = list(realized_graphs.keys())[0]
    sample_graph = realized_graphs[sample_id]
    sample_recipe = sample_graph['recipe_id']

    # Create NetworkX graph
    G = nx.DiGraph()

    nodes = task_graph_embeddings[sample_recipe]['nodes']
    for i, node in enumerate(nodes):
        matched = sample_graph['node_matched'][i]
        sim = sample_graph['match_similarities'][i]
        G.add_node(i, label=node['description'][:20], matched=matched, sim=sim)

    for edge in task_graph_embeddings[sample_recipe]['edges']:
        G.add_edge(edge['from'], edge['to'])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    pos = nx.spring_layout(G, k=2)

    # Color nodes by match status
    matched_nodes = [n for n in G.nodes() if G.nodes[n]['matched']]
    unmatched_nodes = [n for n in G.nodes() if not G.nodes[n]['matched']]

    nx.draw_networkx_nodes(G, pos, nodelist=matched_nodes, node_color='green',
                           node_size=500, alpha=0.7, label='Matched')
    nx.draw_networkx_nodes(G, pos, nodelist=unmatched_nodes, node_color='red',
                           node_size=500, alpha=0.7, label='Unmatched')

    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)

    labels = {n: G.nodes[n]['label'] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title(f"Realized Task Graph: {sample_id}\nRecipe: {sample_recipe}, Label: {'Incorrect' if sample_graph['recipe_label'] else 'Correct'}")
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('extension_data/sample_realized_graph.png', dpi=150)
    plt.show()

except ImportError:
    print("NetworkX not installed. Skip graph visualization.")

# %% [cell 23]
# ## Summary
#
# In this notebook, we completed **Substep 3: Task-Graph Encoding + Step Matching**:
#
# 1. ✅ Built task graphs from step annotations
# 2. ✅ **Encoded task graph nodes using EgoVLP text encoder** (spec-aligned)
#    - Uses `model.compute_text()` for aligned text-visual embeddings
#    - Falls back to sentence-transformers if EgoVLP not available
# 3. ✅ Implemented Hungarian matching between visual and text embeddings
# 4. ✅ Created "realized" task graphs with combined visual-text features
# 5. ✅ Analyzed matching quality
# 6. ✅ Saved realized graphs for GNN classification
#
# **Key Design Choice - EgoVLP Text Encoder**:
# - The project specs require using the EgoVLP/PE textual encoder because the text and video embedding spaces are **jointly trained** and thus **aligned**
# - This means visual step embeddings and text node embeddings can be directly compared using cosine similarity
# - Using a different text encoder (like DistilBERT) would require learning a projection, which is less effective
#
# **Key Observations**:
# - Hungarian matching provides optimal one-to-one assignment
# - Match similarity can indicate recipe correctness
# - Unmatched nodes suggest missing/skipped steps
#
# **Next**: Proceed to **Substep 4** - GNN Classification of observed task-graph
#
# We will train a Graph Neural Network to classify whether the realized task graph represents a correct or incorrect recipe execution.

