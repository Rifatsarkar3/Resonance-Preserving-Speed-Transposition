"""WaPIGT Training Script: Entry point for training with configurable tasks and seeds."""
import argparse
import sys
from pathlib import Path
import yaml
import logging
import torch
from torch.utils.data import DataLoader
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.reproducibility import set_all_seeds
from src.models.wapigt import WaPIGT
from src.models.scr import SpectrumConsistencyRegularizer
from src.training.loss import WaPIGTLoss
from src.training.trainer import WaPIGTTrainer
from src.data_loaders.raw_dataset import RawBearingDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def adjust_config_for_data_size(config: Config, train_size: int, dataset: str) -> Config:
    """
    Adjust hyperparameters based on training data size.

    Implements data-aware adaptive configuration to prevent overfitting on small datasets.
    Multi-tier regularization strategy with progressive aggressiveness.

    Args:
        config: Original Config object
        train_size: Number of training samples
        dataset: Dataset name (for logging)

    Returns:
        Modified Config object with adjusted hyperparameters
    """
    original_state = {
        "dropout": config.model.dropout,
        "gat_dropout": config.model.gat_dropout,
        "weight_decay": config.training.weight_decay,
        "learning_rate": config.training.learning_rate,
    }

    # Multi-tier adaptive regularization strategy
    if train_size < 100:
        # TINY regime: very aggressive regularization for ultra-small datasets (JNU, <100 samples)
        regime = "TINY"
        config.model.dropout = 0.55
        config.model.gat_dropout = 0.55
        config.training.weight_decay = 5e-3
        config.training.learning_rate = 1e-4  # 80% lower LR for extreme stability
        # Extended warmup for small data stability
        if config.training.warmup_epochs < 25:
            config.training.warmup_epochs = 25
    elif train_size < 150:
        # SMALL regime: moderate aggressive regularization (100-150 samples)
        regime = "SMALL"
        config.model.dropout = 0.40
        config.model.gat_dropout = 0.40
        config.training.weight_decay = 1e-3
    elif train_size < 500:
        # MEDIUM regime: slight adjustments (150-500 samples)
        regime = "MEDIUM"
        config.model.dropout = 0.32
        config.training.weight_decay = 2e-4
    else:
        # LARGE regime: baseline (>500 samples)
        regime = "LARGE"
        pass

    # Log adjustments
    logger.info(f"Data-aware config: {regime} regime ({train_size} samples)")

    changed = False
    # Check dropout
    if config.model.dropout != original_state["dropout"]:
        logger.info(f"  dropout: {original_state['dropout']} → {config.model.dropout}")
        changed = True
    # Check gat_dropout
    if config.model.gat_dropout != original_state["gat_dropout"]:
        logger.info(f"  gat_dropout: {original_state['gat_dropout']} → {config.model.gat_dropout}")
        changed = True
    # Check weight_decay
    if config.training.weight_decay != original_state["weight_decay"]:
        logger.info(f"  weight_decay: {original_state['weight_decay']} → {config.training.weight_decay}")
        changed = True
    # Check learning_rate
    if config.training.learning_rate != original_state["learning_rate"]:
        logger.info(f"  learning_rate: {original_state['learning_rate']} → {config.training.learning_rate}")
        changed = True

    if not changed:
        logger.info("  (using baseline config)")

    return config


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = Config.from_yaml(config_path)
    config.validate_paths()
    return config


def get_task_list(dataset: str, task_arg: str) -> list:
    """
    Get list of tasks to run.

    task_arg: "all" for all tasks, or specific task name like "PU_T01"
    """
    # Define all cross-condition tasks per dataset
    task_map = {
        "PU": [f"PU_T{i:02d}" for i in range(1, 13)],  # 12 tasks
        "CWRU": [f"CWRU_T{i:02d}" for i in range(1, 9)],  # 8 tasks (4 loads × 2 transfers)
        "JNU": [f"JNU_T{i:02d}" for i in range(1, 4)],  # 3 tasks (3 speeds)
    }

    if task_arg == "all":
        return task_map.get(dataset, [])
    else:
        return [task_arg]


def create_dataloaders(
    config: Config,
    dataset: str,
    task: str,
    batch_size: int,
    num_workers: int,
):
    """
    Create train/val/test dataloaders for a specific task.

    Returns:
        (train_loader, val_loader, test_loader, n_classes)
    """
    splits_root = Path(config.data.splits_root)

    # Load split manifest
    dataset_abbr = {"PU": "pu", "CWRU": "cwru", "JNU": "jnu"}[dataset]
    splits_file = splits_root / f"{dataset_abbr}_splits.json"

    if not splits_file.exists():
        logger.error(f"Splits file not found: {splits_file}")
        return None, None, None, None

    with open(splits_file) as f:
        splits = json.load(f)

    # Get task-specific splits
    if task not in splits:
        logger.error(f"Task {task} not found in splits")
        return None, None, None, None

    task_splits = splits[task]
    n_classes = task_splits.get("n_classes", 4)

    # Create datasets using raw data
    if dataset == "CWRU":
        raw_root = Path(config.data.cwru_raw_root)
    elif dataset == "PU":
        raw_root = Path(config.data.pu_raw_root)
    else:
        raw_root = Path(config.data.jnu_raw_root)

    # Extract bearing list, speeds, or loads from splits
    train_bearings = task_splits.get("train", [])
    val_bearings = task_splits.get("val", [])
    test_bearings = task_splits.get("test", [])

    train_speeds = task_splits.get("train_speeds", [])
    val_speeds = task_splits.get("val_speeds", [])
    test_speeds = task_splits.get("test_speeds", [])

    train_loads = task_splits.get("train_loads", [])
    val_loads = task_splits.get("val_loads", [])
    test_loads = task_splits.get("test_loads", [])

    # Determine what type of split we're using
    use_speeds = bool(train_speeds)
    use_loads = bool(train_loads)

    # Enable directional augmentation for JNU (bridges train speed → test speed)
    # Other datasets (PU, CWRU) use task-aware augmentation
    train_test_speeds = test_speeds if dataset == "JNU" else None

    train_dataset = RawBearingDataset(
        dataset=dataset,
        raw_root=raw_root,
        bearing_list=train_bearings if train_bearings else None,
        speed_list=train_speeds if use_speeds else None,
        load_list=train_loads if use_loads else None,
        split="train",
        n_samples_per_bearing=20,
        signal_length=12000,
        task_name=task,
        test_speed_list=train_test_speeds,
    )

    val_dataset = RawBearingDataset(
        dataset=dataset,
        raw_root=raw_root,
        bearing_list=val_bearings if val_bearings else None,
        speed_list=val_speeds if use_speeds else None,
        load_list=val_loads if use_loads else None,
        split="val",
        n_samples_per_bearing=10,
        signal_length=12000,
        task_name=None,
    )

    test_dataset = RawBearingDataset(
        dataset=dataset,
        raw_root=raw_root,
        bearing_list=test_bearings if test_bearings else None,
        speed_list=test_speeds if use_speeds else None,
        load_list=test_loads if use_loads else None,
        split="test",
        n_samples_per_bearing=10,
        signal_length=12000,
        task_name=None,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=num_workers > 0 and config.training.persistent_workers,
        prefetch_factor=config.training.prefetch_factor if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=num_workers > 0 and config.training.persistent_workers,
        prefetch_factor=config.training.prefetch_factor if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=num_workers > 0 and config.training.persistent_workers,
        prefetch_factor=config.training.prefetch_factor if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader, n_classes


def train_task(
    config: Config,
    dataset: str,
    task: str,
    seed: int,
    variant: str = "3M",
    device: str = "cuda",
):
    """Train model on a single task."""
    logger.info(f"Training {variant} on {dataset} - {task} (seed={seed})")

    # Set reproducibility
    set_all_seeds(seed)

    # Create dataloaders FIRST to get n_classes
    train_loader, val_loader, test_loader, n_classes = create_dataloaders(
        config=config,
        dataset=dataset,
        task=task,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    if train_loader is None:
        logger.error(f"Failed to create dataloaders for {task}")
        return None

    train_size = len(train_loader.dataset)
    logger.info(f"Train samples: {train_size}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Adjust config based on data size (data-aware adaptive configuration)
    # Special handling: JNU is always small dataset, force TINY regime regardless of loaded samples
    effective_train_size = train_size if dataset != "JNU" else 50
    config = adjust_config_for_data_size(config, effective_train_size, dataset)

    # For TINY regime: use smaller batch size for better per-sample learning
    if effective_train_size < 100:
        original_batch_size = config.training.batch_size
        config.training.batch_size = max(4, config.training.batch_size // 4)
        logger.info(f"TINY regime: batch size {original_batch_size} → {config.training.batch_size}")

    # Create model with correct n_classes
    logger.info("Creating model...")
    model = WaPIGT(
        n_classes=n_classes,
        hidden_dim=config.model.hidden_dim,
        n_encoder_layers=config.model.n_encoder_layers,
        n_heads=config.model.n_heads,
        mlp_dim=config.model.mlp_dim,
        dropout=config.model.dropout,
        n_gat_heads=config.model.n_gat_heads,
        gat_dropout=config.model.gat_dropout,
        filter_order=config.model.filter_order,
        n_frames=config.model.n_frames,
    )
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Create loss function with correct n_classes
    scr = SpectrumConsistencyRegularizer(sigma=2.0)
    loss_fn = WaPIGTLoss(
        n_classes=n_classes,
        scr_module=scr,
        scr_lambda=config.model.scr_lambda,
        scr_warmup_epochs=config.model.scr_warmup_epochs,
        n_epochs=config.training.n_epochs,
    )

    # Create trainer
    task_name = f"{dataset}_{task}_{variant}_seed{seed}"
    trainer = WaPIGTTrainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        device=device,
        task_name=task_name,
    )

    # Training loop
    logger.info("Starting training...")
    for epoch in range(config.training.n_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)

        val_metrics = trainer.evaluate(val_loader)

        # Log
        is_best = val_metrics["accuracy"] > trainer.best_val_metric
        if is_best:
            trainer.best_val_metric = val_metrics["accuracy"]

        if (epoch + 1) % 10 == 0 or is_best:
            logger.info(
                f"Epoch {epoch+1:3d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
                f"F1: {val_metrics['macro_f1']:.4f}"
            )

        # Save checkpoint
        trainer.save_checkpoint(epoch, val_metrics, is_best=is_best)
        trainer.metrics_history["train_loss"].append(train_loss)
        trainer.metrics_history["val_loss"].append(val_metrics["loss"])
        trainer.metrics_history["val_acc"].append(val_metrics["accuracy"])
        trainer.metrics_history["val_f1"].append(val_metrics["macro_f1"])

        # Early stopping enforcement
        if trainer.early_stop:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    logger.info(
        f"Test Results | Loss: {test_metrics['loss']:.4f} "
        f"Accuracy: {test_metrics['accuracy']:.4f} "
        f"Macro-F1: {test_metrics['macro_f1']:.4f}"
    )

    # Add test metrics to history before saving
    if "test_loss" not in trainer.metrics_history:
        trainer.metrics_history["test_loss"] = []
        trainer.metrics_history["test_acc"] = []
        trainer.metrics_history["test_f1"] = []
    trainer.metrics_history["test_loss"].append(test_metrics["loss"])
    trainer.metrics_history["test_acc"].append(test_metrics["accuracy"])
    trainer.metrics_history["test_f1"].append(test_metrics["macro_f1"])

    # Save metrics
    trainer.save_metrics_json()

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train WaPIGT model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument(
        "--dataset",
        type=str,
        default="PU",
        choices=["PU", "CWRU", "JNU"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        help="Task name (e.g., 'PU_T01') or 'all' for all tasks",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--variant",
        type=str,
        default="3M",
        choices=["3M", "5M"],
        help="Model variant",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get task list
    tasks = get_task_list(args.dataset, args.task)
    logger.info(f"Tasks to run: {tasks}")

    # Train each task
    all_results = {}
    for task in tasks:
        result = train_task(
            config=config,
            dataset=args.dataset,
            task=task,
            seed=args.seed,
            variant=args.variant,
            device=args.device,
        )
        if result:
            all_results[task] = result

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Training Summary")
    logger.info("=" * 70)
    for task, metrics in all_results.items():
        logger.info(
            f"{task}: Accuracy={metrics['accuracy']:.4f} "
            f"Macro-F1={metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
