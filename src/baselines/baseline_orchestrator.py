"""Baseline training orchestrator with Optuna tuning and resume support."""
import json
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from dataclasses import asdict

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.utils.config import Config
from src.utils.reproducibility import set_all_seeds
from src.data_loaders.raw_dataset import RawBearingDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BaselineTrainer:
    """Train a baseline model with Optuna hyperparameter tuning."""

    def __init__(self,
                 baseline_name: str,
                 model_class: Callable,
                 config: Config,
                 checkpoint_manager: 'CheckpointManager',
                 n_trials: int = 50,
                 seeds: list = None):
        """Initialize baseline trainer.

        Args:
            baseline_name: Name of baseline (e.g., 'WDCNN', 'TICNN')
            model_class: Model class constructor
            config: Config object
            checkpoint_manager: CheckpointManager for tracking
            n_trials: Number of Optuna trials
            seeds: List of seeds for multi-seed training
        """
        self.baseline_name = baseline_name
        self.model_class = model_class
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.n_trials = n_trials
        self.seeds = seeds or [42, 1337, 2025, 999, 7]

        self.results = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _objective(self, trial: optuna.Trial, dataset: str, task: str,
                   seed: int) -> float:
        """Optuna objective function for a single trial."""
        try:
            # Suggest hyperparameters (baseline-agnostic)
            params = {
                "learning_rate": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            }

            set_all_seeds(seed)

            # Load data
            val_loader, test_loader = self._load_data(dataset, task)
            if val_loader is None:
                return 0.0

            # Create model
            n_classes = 4  # Standard for all datasets
            model = self.model_class(
                signal_length=12000,
                n_classes=n_classes,
                dropout=params["dropout"],
            ).to(self.device)

            # Train
            val_acc = self._train_epoch(
                model, val_loader, test_loader, params, seed
            )

            trial.report(val_acc, step=0)

            if trial.should_prune():
                raise optuna.TrialPruned()

            return val_acc

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            self.checkpoint_manager.mark_trial_failed(
                self.baseline_name, trial.number, str(e)
            )
            raise

    def _load_data(self, dataset: str, task: str):
        """Load validation and test data for a task."""
        try:
            splits_root = Path(self.config.data.splits_root)
            dataset_abbr = {"PU": "pu", "CWRU": "cwru", "JNU": "jnu"}[dataset]
            splits_file = splits_root / f"{dataset_abbr}_splits.json"

            with open(splits_file) as f:
                splits = json.load(f)

            task_splits = splits.get(task)
            if not task_splits:
                logger.warning(f"No splits found for {task}")
                return None, None

            # Determine raw root
            raw_roots = {
                "PU": self.config.data.pu_raw_root,
                "CWRU": self.config.data.cwru_raw_root,
                "JNU": self.config.data.jnu_raw_root,
            }
            raw_root = Path(raw_roots[dataset])

            # Create datasets
            val_dataset = RawBearingDataset(
                dataset=dataset,
                raw_root=raw_root,
                bearing_list=task_splits.get("val", []),
                speed_list=task_splits.get("val_speeds", []),
                load_list=task_splits.get("val_loads", []),
                split="val",
                n_samples_per_bearing=10,
                signal_length=12000,
                task_name=None,
            )

            test_dataset = RawBearingDataset(
                dataset=dataset,
                raw_root=raw_root,
                bearing_list=task_splits.get("test", []),
                speed_list=task_splits.get("test_speeds", []),
                load_list=task_splits.get("test_loads", []),
                split="test",
                n_samples_per_bearing=10,
                signal_length=12000,
                task_name=None,
            )

            val_loader = DataLoader(
                val_dataset, batch_size=32, shuffle=False, num_workers=0
            )
            test_loader = DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=0
            )

            return val_loader, test_loader

        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return None, None

    def _train_epoch(self, model, val_loader, test_loader, params, seed):
        """Train model for one epoch and return validation accuracy."""
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )

        model.train()
        for batch in val_loader:
            signals = batch['signal'].to(self.device)
            labels = batch['label'].to(self.device)

            optimizer.zero_grad()
            logits, _ = model(signals)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                signals = batch['signal'].to(self.device)
                labels = batch['label'].to(self.device)

                logits, _ = model(signals)
                preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0
        logger.info(f"Val Acc: {val_acc:.4f}, Seed: {seed}")

        return val_acc

    def run_trial(self, trial_id: int, dataset: str, task: str) -> bool:
        """Run a single Optuna trial with multi-seed evaluation.

        Returns:
            True if trial succeeded, False otherwise
        """
        # Skip if already completed
        if self.checkpoint_manager.is_trial_completed(self.baseline_name, trial_id):
            logger.info(f"Trial {trial_id} already completed, skipping")
            return True

        logger.info(f"{'='*70}")
        logger.info(f"Baseline: {self.baseline_name}, Trial: {trial_id}, Dataset: {dataset}, Task: {task}")
        logger.info(f"{'='*70}")

        try:
            # Run Optuna study
            sampler = TPESampler(seed=42)
            pruner = MedianPruner()

            study = optuna.create_study(
                sampler=sampler,
                pruner=pruner,
                direction="maximize",
            )

            # Run for each seed
            trial_results = {"trial_id": trial_id, "dataset": dataset, "task": task, "seeds": []}

            for seed in self.seeds:
                def objective(trial):
                    return self._objective(trial, dataset, task, seed)

                study.optimize(objective, n_trials=1, show_progress_bar=False)

                best_trial = study.best_trial
                trial_results["seeds"].append({
                    "seed": seed,
                    "best_params": best_trial.params,
                    "best_value": best_trial.value,
                })

                logger.info(f"  Seed {seed}: Best Acc = {best_trial.value:.4f}")

            # Save trial metrics
            self.checkpoint_manager.save_trial_metrics(
                self.baseline_name, trial_id, trial_results
            )

            # Mark as completed
            self.checkpoint_manager.mark_trial_completed(
                self.baseline_name, trial_id, trial_results
            )

            self.results.append(trial_results)
            return True

        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            self.checkpoint_manager.mark_trial_failed(
                self.baseline_name, trial_id, str(e)
            )
            return False

    def run_all_trials(self, dataset: str, task: str) -> Dict[str, Any]:
        """Run all trials for a baseline, resuming from checkpoint."""
        completed = self.checkpoint_manager.get_completed_trials(self.baseline_name)
        logger.info(f"Starting {self.baseline_name}: {len(completed)}/{self.n_trials} trials completed")

        for trial_id in range(self.n_trials):
            if trial_id in completed:
                logger.info(f"Skipping trial {trial_id} (already completed)")
                continue

            success = self.run_trial(trial_id, dataset, task)
            if not success:
                logger.warning(f"Trial {trial_id} failed, continuing with next trial")

        # Get final progress
        progress = self.checkpoint_manager.get_baseline_progress(
            self.baseline_name, self.n_trials
        )

        logger.info(f"\n{self.baseline_name} Summary:")
        logger.info(f"  Completed: {progress['completed']}/{progress['total']}")
        logger.info(f"  Failed: {progress['failed']}")
        logger.info(f"  Progress: {progress['percent_complete']:.1f}%")

        return progress


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Baseline orchestrator imported successfully")
