# Auto-generated from a Jupyter notebook.
# Source: extension_step1_pipeline.ipynb
#
# Notes:
# - Lines starting with !/%/%% are commented out (IPython-only).
# - Run from the repo root (folder containing requirements.txt).

# %% [cell 0]
# (colab-only setup cell omitted)

# %% [cell 1]
# (colab-only setup cell omitted)

# %% [cell 2]
# ## 1. Load Annotations and Splits

# %% [cell 3]
# Load step annotations
with open(ANNOTATION_PATH, 'r') as f:
    step_annotations = json.load(f)
print(f"Loaded annotations for {len(step_annotations)} recordings")

# Load splits
with open(SPLIT_PATH, 'r') as f:
    splits = json.load(f)
print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

# %% [cell 4]
# ## 2. Feature Loading Utilities

# %% [cell 5]
def load_egovlp_features(recording_id):
    """Load EgoVLP features for a recording."""
    # Local features are stored as .npz with a `video_features` array:
    #   data/features/egovlp/<recording_id>_360p_224.npz
    feature_path = os.path.join(EGOVLP_FEATURE_DIR, f"{recording_id}_360p_224.npz")
    if not os.path.exists(feature_path):
        return None
    data = np.load(feature_path)
    if "video_features" not in data:
        return None
    return data["video_features"]

def pool_features(features, start_time, end_time, fps=0.5):
    """
    Pool features within a time window using mean pooling.
    EgoVLP features in this repo are extracted approximately every ~2 seconds (fps=0.5).
    """
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Clamp to valid range
    start_frame = max(0, start_frame)
    end_frame = min(len(features), end_frame)
    
    if start_frame >= end_frame or start_frame >= len(features):
        return np.zeros(features.shape[1])
    
    return np.mean(features[start_frame:end_frame], axis=0)

# Test loading
test_rec = list(step_annotations.keys())[0]
test_features = load_egovlp_features(test_rec)
if test_features is not None:
    print(f"Test features shape: {test_features.shape}")
else:
    print("Could not load test features - check EGOVLP_FEATURE_DIR path")

# %% [cell 6]
# ## 3. Extract Step Embeddings using GT Boundaries

# %% [cell 7]
def extract_step_embeddings_gt(annotations, feature_loader):
    """
    Extract step-level embeddings using ground truth boundaries.
    
    Returns:
        dict: {recording_id: {
            'step_embeddings': np.array of shape (num_steps, feature_dim),
            'step_labels': list of 0/1 (correct/error),
            'recipe_id': int recipe ID,
            'recipe_label': int 0/1 (recording-level error),
            'descriptions': list of step descriptions,
            'segments': list of (start, end) tuples,
            'step_ids': list of step IDs
        }}
    """
    processed_data = {}
    
    for rec_id, ann in tqdm(annotations.items(), desc="Extracting features"):
        features = feature_loader(rec_id)
        if features is None:
            continue
        
        step_embeddings = []
        step_labels = []
        step_descriptions = []
        step_segments = []
        step_ids = []
        
        for step in ann.get('steps', []):
            # CaptainCook4D step annotations use `start_time` / `end_time` in seconds.
            start_time = step.get('start_time', 0)
            end_time = step.get('end_time', start_time + 1)
            
            # Pool features for this step
            emb = pool_features(features, start_time, end_time)
            step_embeddings.append(emb)
            
            # Get label (0=correct, 1=error based on has_errors field)
            has_error = step.get('has_errors', False)
            label = 1 if has_error else 0
            step_labels.append(label)
            
            # Store metadata
            step_descriptions.append(step.get('description', ''))
            step_segments.append((start_time, end_time))
            step_ids.append(step.get('step_id', -1))
        
        if len(step_embeddings) > 0:
            recipe_id = int(str(rec_id).split('_')[0])
            recipe_label = int(any(step_labels))

            # Keep both legacy keys ('embeddings'/'labels') and the new keys
            # expected by extension_step2_verification_baseline.ipynb.
            step_embeddings_arr = np.array(step_embeddings)
            processed_data[rec_id] = {
                'step_embeddings': step_embeddings_arr,
                'step_labels': step_labels,
                'recipe_id': recipe_id,
                'recipe_label': recipe_label,
                'embeddings': step_embeddings_arr,
                'labels': step_labels,
                'descriptions': step_descriptions,
                'segments': step_segments,
                'step_ids': step_ids,
                'num_steps': len(step_embeddings)
            }
    
    return processed_data

# Extract embeddings
print("Extracting step embeddings using GT boundaries...")
processed_data = extract_step_embeddings_gt(step_annotations, load_egovlp_features)
print(f"Processed {len(processed_data)} recordings")

# %% [cell 8]
# Statistics
total_steps = sum(d['num_steps'] for d in processed_data.values())
total_errors = sum(sum(d['labels']) for d in processed_data.values())
feature_dim = list(processed_data.values())[0]['embeddings'].shape[1]

print(f"\n=== Dataset Statistics ===")
print(f"Total recordings: {len(processed_data)}")
print(f"Total steps: {total_steps}")
print(f"Error steps: {total_errors} ({100*total_errors/total_steps:.1f}%)")
print(f"Correct steps: {total_steps - total_errors} ({100*(total_steps-total_errors)/total_steps:.1f}%)")
print(f"Feature dimension: {feature_dim}")

# %% [cell 9]
# ## 4. Save Output for Pipeline

# %% [cell 10]
# (colab-only setup cell omitted)

# %% [cell 11]
# ## 5. Verify Output

# %% [cell 12]
# Verify the saved file
with open(local_output_path, 'rb') as f:
    loaded = pickle.load(f)

print("=== Verification ===")
print(f"Keys: {loaded.keys()}")
print(f"Recordings: {len(loaded['data'])}")
print(f"Feature dim: {loaded['feature_dim']}")
print(f"Method: {loaded['method']}")
print(f"\nSplits:")
print(f"  Train: {len(loaded['splits']['train'])}")
print(f"  Val: {len(loaded['splits']['val'])}")
print(f"  Test: {len(loaded['splits']['test'])}")

# Sample recording
sample_rec = list(loaded['data'].keys())[0]
sample_data = loaded['data'][sample_rec]
print(f"\nSample recording ({sample_rec}):")
print(f"  Embeddings shape: {sample_data['embeddings'].shape}")
print(f"  Labels: {sample_data['labels'][:5]}...")
print(f"  Descriptions: {sample_data['descriptions'][:2]}...")

# %% [cell 13]
# ## Done!
#
# The step embeddings have been saved. You can now proceed to:
# - **Step 2**: `extension_step2_verification_baseline.ipynb` - Train verification models
# - **Step 3**: `extension_step3_task_graph_matching.ipynb` - Match steps to task graphs
# - **Step 4**: `extension_step4_gnn_classification.ipynb` - GNN classification
#
# For ActionFormer experiments and hyperparameter analysis, see:
# - `extension_step1_actionformer.ipynb`

