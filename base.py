import csv
import os

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import wandb
from torch import optim, nn
from torch.utils.data import DataLoader

from constants import Constants as const
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from torcheval.metrics.functional import binary_auprc
from tqdm import tqdm

from core.models.blocks import fetch_input_dim, MLP
from core.models.er_former import ErFormer
from core.models.lstm import LSTM
from dataloader.CaptainCookStepDataset import collate_fn, CaptainCookStepDataset
from dataloader.CaptainCookSubStepDataset import CaptainCookSubStepDataset


def _metric_value(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _pool_step_probability(step_output: np.ndarray, step_pooling: str, step_topk_frac: float) -> float:
    if step_output.size == 0:
        return 0.0

    if step_pooling == "max":
        return float(np.max(step_output))

    if step_pooling == "topk":
        if step_topk_frac <= 0:
            return float(np.mean(step_output))
        k = max(1, int(np.ceil(step_output.size * step_topk_frac)))
        topk = np.partition(step_output, -k)[-k:]
        return float(np.mean(topk))

    # default: mean
    return float(np.mean(step_output))


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float):
    pred = (y_score > threshold).astype(int)
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    accuracy = accuracy_score(y_true, pred)

    auc = float("nan")
    if np.unique(y_true).size > 1:
        auc = roc_auc_score(y_true, y_score)

    pr_auc = binary_auprc(
        torch.tensor(y_score, dtype=torch.float32),
        torch.tensor(y_true, dtype=torch.long),
    )

    return {
        const.PRECISION: precision,
        const.RECALL: recall,
        const.F1: f1,
        const.ACCURACY: accuracy,
        const.AUC: auc,
        const.PR_AUC: pr_auc,
    }


def fetch_model_name(config):
    if config.task_name == const.ERROR_CATEGORY_RECOGNITION:
        return fetch_model_name_ecr(config)
    elif config.task_name in  [const.EARLY_ERROR_RECOGNITION, const.ERROR_RECOGNITION]:
        if config.model_name is None:
            if config.backbone in [const.RESNET3D, const.X3D, const.SLOWFAST, const.OMNIVORE]:
                config.model_name = f"{config.task_name}_{config.split}_{config.backbone}_{config.variant}_{config.modality[0]}"
            elif config.backbone == const.IMAGEBIND:
                combined_modality_name = '_'.join(config.modality)
                config.model_name = f"{config.task_name}_{config.split}_{config.backbone}_{config.variant}_{combined_modality_name}"
    return config.model_name


def fetch_model_name_ecr(config):
    combined_modality_name = '_'.join(config.modality)
    if config.model_name is None:
        config.model_name = (f"{config.task_name}_{config.split}_{config.backbone}"
                             f"_{config.variant}_{combined_modality_name}_{config.error_category}")
    return config.model_name


def fetch_model(config):
    model = None
    if config.variant == const.MLP_VARIANT:
        if config.backbone in [const.OMNIVORE, const.RESNET3D, const.X3D, const.SLOWFAST, const.IMAGEBIND, const.EGOVLP]:
            input_dim = fetch_input_dim(config)
            model = MLP(input_dim, 512, 1)
    elif config.variant == const.TRANSFORMER_VARIANT:
        if config.backbone in [const.OMNIVORE, const.RESNET3D, const.X3D, const.SLOWFAST, const.IMAGEBIND, const.EGOVLP]:
            model = ErFormer(config)
    elif config.variant == const.LSTM_VARIANT:
        if config.backbone in [const.OMNIVORE, const.RESNET3D, const.X3D, const.SLOWFAST, const.IMAGEBIND, const.EGOVLP]:
            model = LSTM(config)

    assert model is not None, f"Model not found for variant: {config.variant} and backbone: {config.backbone}"
    model.to(config.device)
    return model


def convert_and_round(value):
    value = value * 100.0
    if isinstance(value, torch.Tensor):
        return np.round(value.numpy(), 2)
    return np.round(value, 2)


def collate_stats(config, sub_step_metrics, step_metrics):
    collated_stats = [config.split, config.backbone, config.variant, config.modality]
    for metric in [const.PRECISION, const.RECALL, const.F1, const.ACCURACY, const.AUC, const.PR_AUC]:
        collated_stats.append(convert_and_round(sub_step_metrics[metric]))
    for metric in [const.PRECISION, const.RECALL, const.F1, const.ACCURACY, const.AUC, const.PR_AUC]:
        # Round to two digits before appending
        collated_stats.append(convert_and_round(step_metrics[metric]))
    return collated_stats


def save_results_to_csv(config, sub_step_metrics, step_metrics, step_normalization=False, sub_step_normalization=False,
                        threshold=0.5):
    results_dir = os.path.join(os.getcwd(), const.RESULTS)
    task_results_dir = os.path.join(results_dir, config.task_name, "combined_results")
    os.makedirs(task_results_dir, exist_ok=True)
    config.model_name = fetch_model_name(config)

    results_file_path = os.path.join(task_results_dir,
                                     f'step_{step_normalization}_substep_{sub_step_normalization}_threshold_{threshold}.csv')
    collated_stats = collate_stats(config, sub_step_metrics, step_metrics)

    file_exist = os.path.isfile(results_file_path)

    with open(results_file_path, "a", newline='') as activity_idx_step_idx_annotation_csv_file:
        writer = csv.writer(activity_idx_step_idx_annotation_csv_file, quoting=csv.QUOTE_NONNUMERIC)
        if not file_exist:
            writer.writerow([
                "Split", "Backbone", "Variant", "Modality",
                "Sub-Step Precision", "Sub-Step Recall", "Sub-Step F1", "Sub-Step Accuracy", "Sub-Step AUC",
                "Sub-Step PR AUC",
                "Step Precision", "Step Recall", "Step F1", "Step Accuracy", "Step AUC", "Step PR AUC"
            ])
        writer.writerow(collated_stats)


def save_results(config, sub_step_metrics, step_metrics, step_normalization=False, sub_step_normalization=False,
                 threshold=0.5):
    # 1. Save evaluation results to csv
    save_results_to_csv(config, sub_step_metrics, step_metrics, step_normalization, sub_step_normalization, threshold)


def store_model(model, config, ckpt_name: str):
    task_directory = os.path.join(config.ckpt_directory, config.task_name)
    os.makedirs(task_directory, exist_ok=True)

    variant_directory = os.path.join(task_directory, config.variant)
    os.makedirs(variant_directory, exist_ok=True)

    backbone_directory = os.path.join(variant_directory, config.backbone)
    os.makedirs(backbone_directory, exist_ok=True)

    ckpt_file_path = os.path.join(backbone_directory, ckpt_name)
    torch.save(model.state_dict(), ckpt_file_path)


# ----------------------- TRAIN BASE FILES -----------------------


def train_epoch(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loader = tqdm(train_loader)
    num_batches = len(train_loader)
    train_losses = []

    for batch_idx, batch in enumerate(train_loader):
        data, target = batch[0], batch[1]
        # Handle empty batches (from filtering corrupt files)
        if len(data) == 0:
            continue
            
        data, target = data.to(device), target.to(device)

        # Skip batch if input contains NaNs (prevent contamination)
        if torch.isnan(data).any():
            print(f"Warning: NaNs in input data at batch {batch_idx}. Skipping.")
            continue

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if torch.isnan(loss).any():
             print(f"Warning: NaN loss detected at batch {batch_idx}. Skipping step.")
             continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        train_losses.append(loss.item())
        train_loader.set_description(
            f'Train Epoch: {epoch}, Progress: {batch_idx}/{num_batches}, Loss: {loss.item():.6f}'
        )

    return train_losses


def train_model_base(train_loader, val_loader, config, test_loader=None):
    model = fetch_model(config)
    device = config.device
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Use pos_weight from config if available, else default to 2.5
    pos_weight_val = getattr(config, 'pos_weight', 2.5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32).to(device))
    
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max',
        factor=0.1, patience=5,
        threshold=1e-4, threshold_mode="abs", min_lr=1e-7
    )
    # criterion = nn.BCEWithLogitsLoss()
    # Track best checkpoint using the chosen metric (default: step F1)
    best_model = {'model_state': None, 'metric': float("-inf"), 'epoch': None}
    no_improve_epochs = 0

    model_name = config.model_name
    if config.model_name is None:
        model_name = fetch_model_name(config)
        config.model_name = model_name

    train_stats_directory = f"stats/{config.task_name}/{config.variant}/{config.backbone}"
    os.makedirs(train_stats_directory, exist_ok=True)
    train_stats_file = f"{model_name}_training_performance.txt"
    train_stats_file_path = os.path.join(train_stats_directory, train_stats_file)

    # Open a file to store the losses and metrics
    with open(train_stats_file_path, 'w') as f:
        f.write('Epoch, Train Loss, Test Loss, Precision, Recall, F1, AUC\n')
        for epoch in range(1, config.num_epochs + 1):

            model.train()
            train_loader = tqdm(train_loader)
            num_batches = len(train_loader)
            train_losses = []

            for batch_idx, batch in enumerate(train_loader):
                data, target = batch[0], batch[1]
                # Robustness: Skip empty batches (Edit 004)
                if len(data) == 0:
                    continue

                data, target = data.to(device), target.to(device)

                # Robustness: Skip corrupt input (Edit 004)
                if torch.isnan(data).any():
                    # print(f"Warning: NaNs in input data. Skipping.")
                    continue

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)

                if torch.isnan(loss).any():
                     # Just skip silently or with minimal logging to avoid spam
                     # print(f"Warning: NaN loss ignored.")
                     optimizer.zero_grad()
                     continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                train_losses.append(loss.item())
                train_loader.set_description(
                    f'Train Epoch: {epoch}, Progress: {batch_idx}/{num_batches}, Loss: {loss.item():.6f}'
                )

            # Use threshold from config
            eval_threshold = getattr(config, 'threshold', 0.6)
            val_losses, sub_step_metrics, step_metrics = test_er_model(
                model,
                val_loader,
                criterion,
                device,
                phase='val',
                threshold=eval_threshold,
                step_pooling=getattr(config, "step_pooling", "mean"),
                step_topk_frac=getattr(config, "step_topk_frac", 0.2),
                sweep_thresholds=getattr(config, "sweep_thresholds", False),
                sweep_min=getattr(config, "sweep_min", 0.1),
                sweep_max=getattr(config, "sweep_max", 0.9),
                sweep_step=getattr(config, "sweep_step", 0.05),
            )

            scheduler.step(step_metrics[const.AUC])

            eval_threshold_for_test = eval_threshold
            if getattr(config, "sweep_thresholds", False):
                eval_threshold_for_test = float(step_metrics.get("best_threshold", eval_threshold))

            if test_loader is not None:
                test_losses, test_sub_step_metrics, test_step_metrics = test_er_model(
                    model,
                    test_loader,
                    criterion,
                    device,
                    phase='test',
                    threshold=eval_threshold_for_test,
                    step_pooling=getattr(config, "step_pooling", "mean"),
                    step_topk_frac=getattr(config, "step_topk_frac", 0.2),
                    sweep_thresholds=False,
                    sweep_min=getattr(config, "sweep_min", 0.1),
                    sweep_max=getattr(config, "sweep_max", 0.9),
                    sweep_step=getattr(config, "sweep_step", 0.05),
                )

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_test_loss = sum(test_losses) / len(test_losses)

            precision = step_metrics[const.PRECISION]
            recall = step_metrics[const.RECALL]
            f1 = step_metrics[const.F1]
            auc = step_metrics[const.AUC]

            # Write losses and metrics to file
            f.write(
                f'{epoch}, {avg_train_loss:.6f}, {avg_val_loss:.6f}, {avg_test_loss:.6f}, {precision:.6f}, {recall:.6f}, {f1:.6f}, {auc:.6f}\n')

            running_metrics = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "test_loss": avg_test_loss,
                "val_loss": avg_val_loss,
                "val_metrics": {
                    "step_metrics": step_metrics,
                    "sub_step_metrics": sub_step_metrics
                },
                "test_metrics": {
                    "step_metrics": test_step_metrics,
                    "sub_step_metrics": test_sub_step_metrics
                }
            }

            if config.enable_wandb:
                wandb.log(running_metrics)

            print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, '
                  f'Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}, AUC: {auc:.6f}')

            # Update best model based on configured selection metric (default: F1)
            best_metric_name = getattr(config, "best_metric", "f1")
            if best_metric_name == "auc":
                current_metric = _metric_value(step_metrics[const.AUC])
            elif best_metric_name == "pr_auc":
                current_metric = _metric_value(step_metrics[const.PR_AUC])
            else:
                current_metric = _metric_value(step_metrics[const.F1])

            if current_metric > best_model['metric']:
                best_model['metric'] = current_metric
                best_model['model_state'] = model.state_dict()
                best_model['epoch'] = epoch
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            early_stop_patience = int(getattr(config, "early_stop_patience", 0) or 0)
            if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_model['epoch']}, {best_metric_name}={best_model['metric']:.6f}).")
                break

            store_model(model, config, ckpt_name=f"{model_name}_epoch_{epoch}.pt")

        # Save the best model
        if best_model['model_state'] is not None:
            model.load_state_dict(best_model['model_state'])
            store_model(model, config, ckpt_name=f"{model_name}_best.pt")


def train_step_test_step_dataset_base(config):
    torch.manual_seed(config.seed)

    cuda_kwargs = {
        "num_workers": 0,
        "pin_memory": False,
    }
    train_kwargs = {**cuda_kwargs, "shuffle": True, "batch_size": config.batch_size}
    test_kwargs = {**cuda_kwargs, "shuffle": False, "batch_size": config.test_batch_size}

    print("-------------------------------------------------------------")
    print("Training step model and testing on step level")
    print(f"Train args: {train_kwargs}")
    print(f"Test args: {test_kwargs}")
    if config.error_category is not None:
        print(f"Error Category: {config.error_category}")
    print(config.args)
    print("-------------------------------------------------------------")

    train_dataset = CaptainCookStepDataset(config, const.TRAIN, config.split)
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, **train_kwargs)
    val_dataset = CaptainCookStepDataset(config, const.VAL, config.split)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, **test_kwargs)
    test_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, **test_kwargs)

    return train_loader, val_loader, test_loader


def train_sub_step_test_step_dataset_base(config):
    torch.manual_seed(config.seed)

    cuda_kwargs = {
        "num_workers": 1,
        "pin_memory": False,
    }
    train_kwargs = {**cuda_kwargs, "shuffle": True, "batch_size": 1024}
    test_kwargs = {**cuda_kwargs, "shuffle": False, "batch_size": 1}

    train_dataset = CaptainCookSubStepDataset(config, const.TRAIN, config.split)
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, **train_kwargs)
    val_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, **test_kwargs)
    test_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, **test_kwargs)

    print("-------------------------------------------------------------")
    print("Training sub-step model and testing on step level")
    print(f"Train args: {train_kwargs}")
    print(f"Test args: {test_kwargs}")
    print(f"Split: {config.split}")
    print("-------------------------------------------------------------")

    return train_loader, val_loader, test_loader


# ----------------------- TEST BASE FILES -----------------------


def test_er_model(model, test_loader, criterion, device, phase, step_normalization=True, sub_step_normalization=True,
                  threshold=0.6, step_pooling="mean", step_topk_frac=0.2, sweep_thresholds=False, sweep_min=0.1,
                  sweep_max=0.9, sweep_step=0.05):
    total_samples = 0
    all_targets = []
    all_outputs = []
    all_step_lengths = []

    test_loader = tqdm(test_loader)
    num_batches = len(test_loader)
    test_losses = []

    counter = 0

    with torch.no_grad():
        for batch in test_loader:
            data, target = batch[0], batch[1]
            step_lengths = batch[2] if len(batch) > 2 else None
              # Robustness: Skip empty batches
            if len(data) == 0:
                continue

            data, target = data.to(device), target.to(device)
            
            # Robustness: Skip corrupt input
            if torch.isnan(data).any():
                continue

            output = model(data)
            total_samples += data.shape[0]
            loss = criterion(output, target)
            
            # Robustness: Handle NaN loss in testing
            loss_val = loss.item()
            if np.isnan(loss_val):
                continue
                
            test_losses.append(loss_val)

            sigmoid_output = output.sigmoid()
            all_outputs.append(sigmoid_output.detach().cpu().numpy().reshape(-1))
            all_targets.append(target.detach().cpu().numpy().reshape(-1))

            if step_lengths is not None:
                all_step_lengths.extend(step_lengths)
            counter += int(data.shape[0])

            # Set the description of the tqdm instance to show the loss
            test_loader.set_description(f'{phase} Progress: {total_samples}/{num_batches}')

    # Flatten lists
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    # Assert that none of the outputs are NaN
    assert not np.isnan(all_outputs).any(), "Outputs contain NaN values"

    # ------------------------- Sub-Step Level Metrics -------------------------
    all_sub_step_targets = all_targets.copy()
    all_sub_step_outputs = all_outputs.copy()

    # Calculate metrics at the sub-step level
    sub_step_metrics = _binary_metrics(all_sub_step_targets, all_sub_step_outputs, threshold=0.5)

    # -------------------------- Step Level Metrics --------------------------
    all_step_targets = []
    all_step_outputs = []

    # threshold_outputs = all_outputs / max_probability

    if not all_step_lengths:
        raise RuntimeError(
            "Step-level metrics require per-step boundaries, but none were provided. "
            "Ensure `dataloader/CaptainCookStepDataset.py:collate_fn` returns step lengths."
        )

    offset = 0
    for step_len in all_step_lengths:
        start, end = offset, offset + int(step_len)
        step_output = all_outputs[start:end]
        step_target = all_targets[start:end]
        offset = end

        # sorted_step_output = np.sort(step_output)
        # # Top 50% of the predictions
        # threshold = np.percentile(sorted_step_output, 50)
        # step_output = step_output[step_output > threshold]

        # pos_output = step_output[step_output > 0.5]
        # neg_output = step_output[step_output <= 0.5]
        #
        # if len(pos_output) > len(neg_output):
        #     step_output = pos_output
        # else:
        #     step_output = neg_output
        if len(step_output) == 0:
            # Handle empty steps (robustness)
            mean_step_output = 0.0 
            step_target = 0
            # Optional: Log warning if needed, but safe default avoids crash
            # print(f"Warning: Empty step encountered at index {start}:{end}")
        else:
            step_output = np.array(step_output)
            # # Scale the output to [0, 1]
            if end - start > 1:
                if sub_step_normalization:
                    prob_range = np.max(step_output) - np.min(step_output)
                    step_output = (step_output - np.min(step_output)) / prob_range

            mean_step_output = _pool_step_probability(
                step_output,
                step_pooling=step_pooling,
                step_topk_frac=step_topk_frac,
            )
            step_target = 1 if np.mean(step_target) > 0.5 else 0

        all_step_outputs.append(mean_step_output)
        all_step_targets.append(step_target)

    all_step_outputs = np.array(all_step_outputs)
    
    # Remove any NaNs that might have slipped through
    # (though the check above should handle empty steps)
    if np.isnan(all_step_outputs).any():
        print("Warning: NaNs detected in all_step_outputs. Replacing with 0.")
        all_step_outputs = np.nan_to_num(all_step_outputs, nan=0.0)

    # # Scale the output to [0, 1]
    if step_normalization and len(all_step_outputs) > 0:
        prob_range = np.max(all_step_outputs) - np.min(all_step_outputs)
        if prob_range > 1e-9: # Avoid division by zero
             all_step_outputs = (all_step_outputs - np.min(all_step_outputs)) / prob_range

    all_step_targets = np.array(all_step_targets)

    # Calculate metrics at the step level (single threshold)
    step_metrics = _binary_metrics(all_step_targets, all_step_outputs, threshold=threshold)

    # Threshold sweep (largest practical gain: pick threshold that maximizes Step-Level F1)
    if sweep_thresholds:
        thresholds = np.arange(float(sweep_min), float(sweep_max) + 1e-9, float(sweep_step))
        best_key = None
        best_threshold = float(threshold)
        best_metrics = step_metrics
        for t in thresholds:
            metrics_t = _binary_metrics(all_step_targets, all_step_outputs, threshold=float(t))
            key = (_metric_value(metrics_t[const.F1]), _metric_value(metrics_t[const.PRECISION]))
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = float(t)
                best_metrics = metrics_t

        step_metrics = best_metrics
        step_metrics["best_threshold"] = best_threshold
        step_metrics["sweep_min"] = float(sweep_min)
        step_metrics["sweep_max"] = float(sweep_max)
        step_metrics["sweep_step"] = float(sweep_step)

    # Print step level metrics
    print("----------------------------------------------------------------")
    print(f'{phase} Sub Step Level Metrics: {sub_step_metrics}')
    print(f"{phase} Step Level Metrics: {step_metrics}")
    print("----------------------------------------------------------------")

    return test_losses, sub_step_metrics, step_metrics
