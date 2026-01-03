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
from dataloader.CaptainCookStepDataset import collate_fn, CaptainCookStepDataset
from dataloader.CaptainCookSubStepDataset import CaptainCookSubStepDataset


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
        if config.backbone in [const.OMNIVORE, const.RESNET3D, const.X3D, const.SLOWFAST, const.IMAGEBIND]:
            input_dim = fetch_input_dim(config)
            model = MLP(input_dim, 512, 1)
    elif config.variant == const.TRANSFORMER_VARIANT:
        if config.backbone in [const.OMNIVORE, const.RESNET3D, const.X3D, const.SLOWFAST, const.IMAGEBIND]:
            model = ErFormer(config)

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

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        assert not torch.isnan(data).any(), "Data contains NaN values"

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        assert not torch.isnan(loss).any(), "Loss contains NaN values"

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
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5], dtype=torch.float32).to(device))
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max',
        factor=0.1, patience=5, verbose=True,
        threshold=1e-4, threshold_mode="abs", min_lr=1e-7
    )
    # criterion = nn.BCEWithLogitsLoss()
    # Initialize variables to track the best model based on the desired metric (e.g., AUC)
    best_model = {'model_state': None, 'metric': 0}

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

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                assert not torch.isnan(data).any(), "Data contains NaN values"

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)

                if torch.isnan(loss).any():
                    print(f"Loss contains NaN values in epoch {epoch}, batch {batch_idx}")
                    continue

                # assert not torch.isnan(loss).any(), "Loss contains NaN values"

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                train_losses.append(loss.item())
                train_loader.set_description(
                    f'Train Epoch: {epoch}, Progress: {batch_idx}/{num_batches}, Loss: {loss.item():.6f}'
                )

            val_losses, sub_step_metrics, step_metrics = test_er_model(model, val_loader, criterion, device, phase='val')

            scheduler.step(step_metrics[const.AUC])

            if test_loader is not None:
                test_losses, test_sub_step_metrics, test_step_metrics = test_er_model(model, test_loader, criterion,
                                                                                      device, phase='test')

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_test_loss = sum(test_losses) / len(test_losses)

            precision = step_metrics['precision']
            recall = step_metrics['recall']
            f1 = step_metrics['f1']
            auc = step_metrics['auc']

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

            # Update best model based on the chosen metric, here using AUC as an example
            if auc > best_model['metric']:
                best_model['metric'] = auc
                best_model['model_state'] = model.state_dict()

            store_model(model, config, ckpt_name=f"{model_name}_epoch_{epoch}.pt")

        # Save the best model
        if best_model['model_state'] is not None:
            model.load_state_dict(best_model['model_state'])
            store_model(model, config, ckpt_name=f"{model_name}_best.pt")


def train_step_test_step_dataset_base(config):
    torch.manual_seed(config.seed)

    cuda_kwargs = {
        "num_workers": 8,
        "pin_memory": False,
    }
    train_kwargs = {**cuda_kwargs, "shuffle": True, "batch_size": config.batch_size}
    test_kwargs = {**cuda_kwargs, "shuffle": False, "batch_size": 1}

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


def test_er_model(
    model,
    test_loader,
    criterion,
    device,
    phase,
    step_normalization=True,
    sub_step_normalization=True,
    threshold=0.6,
):
    import numpy as np
    import torch
    from collections import defaultdict
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        accuracy_score,
    )
    from tqdm import tqdm

    total_samples = 0
    all_targets = []
    all_outputs = []

    test_loader = tqdm(test_loader)
    num_batches = len(test_loader)
    test_losses = []

    test_step_start_end_list = []
    step_keys = []  # (recording_id, step_id) aligned with test_step_start_end_list
    counter = 0

    # Try to access the dataset's step->error_labels mapping
    ds = getattr(test_loader, "iterable", None)  # tqdm wraps; not always present
    ds = getattr(test_loader, "dataset", None) or getattr(getattr(test_loader, "iterable", None), "dataset", None)
    recording_step_error_labels = getattr(ds, "_recording_step_error_labels", None)

    with torch.no_grad():
        for data, target, recording_ids, step_ids in test_loader:
            # This evaluation code assumes batch_size=1 (otherwise step boundaries are lost by collate_fn's cat()).
            if isinstance(step_ids, (tuple, list)) and len(step_ids) != 1:
                raise ValueError(
                    "test_er_model assumes batch_size=1 for step-level aggregation. "
                    "Your collate_fn concatenates variable-length steps, so batch_size>1 loses boundaries."
                )

            recording_id = recording_ids[0] if isinstance(recording_ids, (tuple, list)) else recording_ids
            step_id = step_ids[0] if isinstance(step_ids, (tuple, list)) else step_ids

            data, target = data.to(device), target.to(device)
            output = model(data)
            total_samples += data.shape[0]

            loss = criterion(output, target)
            test_losses.append(loss.item())

            sigmoid_output = output.sigmoid()
            all_outputs.append(sigmoid_output.detach().cpu().numpy().reshape(-1))
            all_targets.append(target.detach().cpu().numpy().reshape(-1))

            test_step_start_end_list.append((counter, counter + data.shape[0]))
            step_keys.append((recording_id, step_id))
            counter += data.shape[0]

            test_loader.set_description(f"{phase} Progress: {total_samples}/{num_batches}")

    # Flatten
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    assert not np.isnan(all_outputs).any(), "Outputs contain NaN values"

    # ------------------------- Sub-Step Level Metrics -------------------------
    all_sub_step_targets = all_targets.copy()
    all_sub_step_outputs = all_outputs.copy()

    pred_sub_step_labels = (all_sub_step_outputs > 0.5).astype(int)
    sub_step_metrics = {
        const.PRECISION: precision_score(all_sub_step_targets, pred_sub_step_labels, zero_division=0),
        const.RECALL: recall_score(all_sub_step_targets, pred_sub_step_labels, zero_division=0),
        const.F1: f1_score(all_sub_step_targets, pred_sub_step_labels, zero_division=0),
        const.ACCURACY: accuracy_score(all_sub_step_targets, pred_sub_step_labels),
        const.AUC: roc_auc_score(all_sub_step_targets, all_sub_step_outputs)
        if len(np.unique(all_sub_step_targets)) == 2
        else None,
        const.PR_AUC: binary_auprc(torch.tensor(pred_sub_step_labels), torch.tensor(all_sub_step_targets)),
    }

    # -------------------------- Step Level Metrics --------------------------
    all_step_targets = []
    all_step_outputs = []
    all_step_error_sets = []  # set of error_category_idx for that step (possibly multiple)

    for (start, end), (recording_id, step_id) in zip(test_step_start_end_list, step_keys):
        step_output = np.array(all_outputs[start:end])
        step_target = np.array(all_targets[start:end])

        # Normalize within the step (only if more than 1 sub-step)
        if (end - start) > 1 and sub_step_normalization:
            prob_range = np.max(step_output) - np.min(step_output)
            if prob_range > 0:
                step_output = (step_output - np.min(step_output)) / prob_range

        mean_step_output = float(np.mean(step_output))
        step_target_bin = 1 if float(np.mean(step_target)) > 0.95 else 0

        all_step_outputs.append(mean_step_output)
        all_step_targets.append(step_target_bin)

        # Error-type labels (only meaningful for error steps; keep empty set for non-error)
        if step_target_bin == 1 and recording_step_error_labels is not None:
            err_set = recording_step_error_labels.get(recording_id, {}).get(step_id, set())
            all_step_error_sets.append(set(err_set))
        else:
            all_step_error_sets.append(set())

    all_step_outputs = np.array(all_step_outputs, dtype=float)
    all_step_targets = np.array(all_step_targets, dtype=int)

    # Normalize across steps
    if step_normalization:
        prob_range = np.max(all_step_outputs) - np.min(all_step_outputs)
        if prob_range > 0:
            all_step_outputs = (all_step_outputs - np.min(all_step_outputs)) / prob_range

    pred_step_labels = (all_step_outputs > threshold).astype(int)
    step_metrics = {
        const.PRECISION: precision_score(all_step_targets, pred_step_labels, zero_division=0),
        const.RECALL: recall_score(all_step_targets, pred_step_labels, zero_division=0),
        const.F1: f1_score(all_step_targets, pred_step_labels, zero_division=0),
        const.ACCURACY: accuracy_score(all_step_targets, pred_step_labels),
        const.AUC: roc_auc_score(all_step_targets, all_step_outputs) if len(np.unique(all_step_targets)) == 2 else None,
        const.PR_AUC: binary_auprc(torch.tensor(pred_step_labels), torch.tensor(all_step_targets)),
    }

    # -------------------- Error-Type Analysis (one-vs-rest) --------------------
    # For each error category t:
    #   y_true_t = 1 if step has error type t else 0
    #   y_score  = model's step probability (same as baseline)
    per_type_metrics = {}
    if recording_step_error_labels is not None:
        # Collect which type IDs exist in the data
        all_type_ids = sorted({t for s in all_step_error_sets for t in s})

        for t in all_type_ids:
            y_true_t = np.array([1 if (t in s) else 0 for s in all_step_error_sets], dtype=int)
            y_pred_t = (all_step_outputs > threshold).astype(int)

            per_type_metrics[t] = {
                "n_pos": int(y_true_t.sum()),
                "n_total": int(len(y_true_t)),
                "precision": precision_score(y_true_t, y_pred_t, zero_division=0),
                "recall": recall_score(y_true_t, y_pred_t, zero_division=0),
                "f1": f1_score(y_true_t, y_pred_t, zero_division=0),
                "accuracy": accuracy_score(y_true_t, y_pred_t),
                "auc": roc_auc_score(y_true_t, all_step_outputs) if len(np.unique(y_true_t)) == 2 else None,
            }

    print("----------------------------------------------------------------")
    print(f"{phase} Sub Step Level Metrics: {sub_step_metrics}")
    print(f"{phase} Step Level Metrics: {step_metrics}")
    if per_type_metrics:
        print(f"{phase} Error-Type Metrics (one-vs-rest, using step scores):")
        for t, m in per_type_metrics.items():
            print(f"  type={t}: {m}")
    print("----------------------------------------------------------------")

    return test_losses, sub_step_metrics, step_metrics, per_type_metrics
