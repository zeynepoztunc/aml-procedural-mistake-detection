
import os
import torch
import numpy as np
import constants as const
from core.config import Config
from base import fetch_model, test_er_model, train_step_test_step_dataset_base
from torch import nn

def evaluate():
    # 1. Load Config
    config = Config()
    
    # 2. Force Batch Size 1 for evaluation correctness
    config.test_batch_size = 1
    
    # 3. Setup Environment
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"
        
    print(f"Evaluating with Device: {config.device}")
    
    # 4. Load Data
    # train_step_test_step_dataset_base returns (train, val, test) loaders
    _, val_loader, test_loader = train_step_test_step_dataset_base(config)
    
    # 5. Load Model Architecture
    model = fetch_model(config)
    
    # 6. Load Weights
    # Construct path: checkpoints/error_recognition/Transformer/egovlp/None_best.pt
    # Note: 'None' comes from config.model_name not being set initially, assuming defaults similar to training
    
    # We try to find the best model file dynamically
    ckpt_dir = os.path.join(config.ckpt_directory, config.task_name, config.variant, config.backbone)
    
    # Try finding the best model
    # The training script names it f"{model_name}_best.pt"
    # If model_name was None in config, fetch_model_name sets it.
    
    # Hardcoding the name seen in the user's previous logs "None_training_performance.txt" implies model_name might be "None" 
    # OR fetch_model_name returns something that includes "None" if features are missing?
    # Let's list the directory to be safe.
    
    potential_models = [f for f in os.listdir(ckpt_dir) if f.endswith("_best.pt")]
    if not potential_models:
        print(f"No best.pt model found in {ckpt_dir}. Checking for epoch checkpoints...")
        potential_models = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    
    if not potential_models:
        print("No checkpoints found.")
        return

    # Pick the most likely candidate (or the most recent)
    ckpt_path = os.path.join(ckpt_dir, potential_models[0])
    print(f"Loading checkpoint: {ckpt_path}")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=config.device))
    model.eval()
    
    # 7. Run Evaluation
    criterion = nn.BCEWithLogitsLoss()
    
    print("\n--- Validation Set Evaluation ---")
    test_er_model(model, val_loader, criterion, config.device, phase='val')
    
    print("\n--- Test Set Evaluation ---")
    test_er_model(model, test_loader, criterion, config.device, phase='test')

if __name__ == "__main__":
    evaluate()
