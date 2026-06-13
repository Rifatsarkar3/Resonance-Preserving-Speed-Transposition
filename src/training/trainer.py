"""WaPIGT Trainer: Training loop with AMP, checkpointing, and evaluation."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
import json
from typing import Dict, Tuple
import numpy as np
from datetime import datetime
import logging

from src.utils.config import Config


logger = logging.getLogger(__name__)


class WaPIGTTrainer:
    """
    Trainer for WaPIGT model with mixed precision (bf16), checkpointing, and evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: Config,
        device: str = "cuda",
        task_name: str = "default",
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.task_name = task_name
        self.global_step = 0
        self.best_val_metric = -float("inf")

        # Early stopping tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.early_stop = False

        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Setup scheduler with warmup
        self.scheduler = self._setup_scheduler()

        # Setup mixed precision scaler
        self.use_amp = config.training.precision == "bf16"
        if self.use_amp:
            # bf16 doesn't use GradScaler, but we keep the pattern for fp16 compatibility
            self.scaler = None
        else:
            self.scaler = GradScaler()

        # Setup checkpointing
        self.checkpoint_dir = Path(config.training.checkpoint_dir) / task_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }

    def _setup_scheduler(self):
        """Setup cosine annealing scheduler with warmup."""
        warmup_epochs = self.config.training.warmup_epochs
        total_epochs = self.config.training.n_epochs

        # Linear warmup followed by cosine annealing
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        from torch.optim.lr_scheduler import LambdaLR

        return LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            avg_loss: Average training loss
            avg_acc: Average training accuracy
        """
        self.model.train()
        self.loss_fn.set_epoch(len(self.metrics_history["train_loss"]))

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()

            # Move batch to device
            signals = batch["signal"].to(self.device)
            labels = batch["label"].to(self.device)

            # Reorganize bearing_params from {key: [batch values]} to [{key: value}, ...]
            batch_size = signals.shape[0]
            bearing_params_list = []

            # bearing_params is {key: [batch values]} from DataLoader
            bearing_params = batch["bearing_params"]

            # Convert to list of dicts: [{key: val, ...}, ...]
            for i in range(batch_size):
                params_dict = {}
                for key, val in bearing_params.items():
                    if isinstance(val, torch.Tensor):
                        # All values are batched tensors from DataLoader
                        if val.dim() > 0:
                            scalar_val = val[i].item()
                        else:
                            scalar_val = val.item()
                        params_dict[key] = scalar_val
                    elif isinstance(val, (list, tuple)):
                        params_dict[key] = val[i]
                    else:
                        # Scalar value (same for all samples)
                        params_dict[key] = val
                bearing_params_list.append(params_dict)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(dtype=torch.bfloat16):
                    logits, attn_weights, embeddings = self.model(signals, bearing_params_list)
                    fault_freq_bins = batch.get("fault_freq_bins", None)
                    if fault_freq_bins is not None:
                        fault_freq_bins = fault_freq_bins.to(self.device)
                    loss = self.loss_fn(
                        logits,
                        labels,
                        attn_weights,
                        fault_freq_bins,
                        window_len=signals.shape[-1],
                        fs_sampling=batch.get("fs_sampling", 12000.0),
                        embeddings=embeddings,
                    )
            else:
                logits, attn_weights, embeddings = self.model(signals, bearing_params_list)

                fault_freq_bins = batch.get("fault_freq_bins", None)
                if fault_freq_bins is not None:
                    fault_freq_bins = fault_freq_bins.to(self.device)
                loss = self.loss_fn(
                    logits,
                    labels,
                    attn_weights,
                    fault_freq_bins,
                    window_len=signals.shape[-1],
                    fs_sampling=batch.get("fs_sampling", 12000.0),
                    embeddings=embeddings,
                )

            # Trap NaN loss before backward pass
            if torch.isnan(loss):
                logger.error(f"NaN Loss detected at Step {self.global_step}!")
                if fault_freq_bins is not None:
                    logger.error(f"Fault Bins: {fault_freq_bins}")
                logger.error(f"Logits contains NaN: {torch.isnan(logits).any()}")
                logger.error(f"Logits min/max: {logits.min()}/{logits.max()}")
                logger.error(f"Labels: {labels}")
                logger.error(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
                self.optimizer.zero_grad()
                del loss, logits, attn_weights, embeddings, labels, signals
                continue

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.gradient_clip_norm
            )

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Metrics (compute before deleting)
            loss_val = loss.item()
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
                total_correct += correct
                total_samples += labels.shape[0]
                total_loss += loss_val

            # Clean up computation graph to prevent memory accumulation
            del loss, logits, attn_weights, embeddings, labels, signals

            self.global_step += 1

        self.scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        avg_acc = total_correct / max(total_samples, 1)

        return avg_loss, avg_acc

    @torch.no_grad()
    def evaluate(self, val_loader) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Returns:
            Dict with loss, accuracy, macro-F1 score
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in val_loader:
            signals = batch["signal"].to(self.device)
            labels = batch["label"].to(self.device)

            # Reorganize bearing_params from {key: [batch values]} to [{key: value}, ...]
            batch_size = signals.shape[0]
            bearing_params_list = []

            # bearing_params is {key: [batch values]} from DataLoader
            bearing_params = batch["bearing_params"]

            # Convert to list of dicts: [{key: val, ...}, ...]
            for i in range(batch_size):
                params_dict = {}
                for key, val in bearing_params.items():
                    if isinstance(val, torch.Tensor):
                        # All values are batched tensors from DataLoader
                        if val.dim() > 0:
                            scalar_val = val[i].item()
                        else:
                            scalar_val = val.item()
                        params_dict[key] = scalar_val
                    elif isinstance(val, (list, tuple)):
                        params_dict[key] = val[i]
                    else:
                        # Scalar value (same for all samples)
                        params_dict[key] = val
                bearing_params_list.append(params_dict)

            # Get attention weights and fault_freq_bins for loss computation
            fault_freq_bins = batch.get("fault_freq_bins", None)
            if fault_freq_bins is not None:
                fault_freq_bins = fault_freq_bins.to(self.device)

            if self.use_amp:
                with autocast(dtype=torch.bfloat16):
                    logits, attn_weights, _ = self.model(signals, bearing_params_list)
                    loss = self.loss_fn(
                        logits,
                        labels,
                        attn_weights,
                        fault_freq_bins,
                        window_len=signals.shape[-1],
                        fs_sampling=batch.get("fs_sampling", 12000.0),
                    )
            else:
                logits, attn_weights, _ = self.model(signals, bearing_params_list)
                loss = self.loss_fn(
                    logits,
                    labels,
                    attn_weights,
                    fault_freq_bins,
                    window_len=signals.shape[-1],
                    fs_sampling=batch.get("fs_sampling", 12000.0),
                )

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            del logits, loss, signals, labels, attn_weights

        avg_loss = total_loss / max(len(val_loader), 1)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        accuracy = (all_preds == all_labels).mean()

        # Compute macro-F1
        from sklearn.metrics import f1_score

        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # Check early stopping condition (val loss divergence)
        self._check_early_stopping(avg_loss)

        return {"loss": avg_loss, "accuracy": accuracy, "macro_f1": macro_f1}

    def _check_early_stopping(self, val_loss: float, patience: int = 150, divergence_threshold: float = 5.0):
        """Stop early if validation loss diverges or plateaus."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Stop if val loss grows > 50% above best loss (divergence)
        if val_loss > self.best_val_loss * divergence_threshold:
            self.early_stop = True
            logger.warning(f"Early stopping: val_loss {val_loss:.4f} > {divergence_threshold}x best {self.best_val_loss:.4f}")
        # Stop if no improvement for 'patience' epochs
        elif self.patience_counter >= patience:
            self.early_stop = True
            logger.warning(f"Early stopping: no improvement for {patience} epochs")

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "global_step": self.global_step,
        }

        # Save last checkpoint
        last_path = self.checkpoint_dir / "last.pt"
        torch.save(checkpoint, last_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint at epoch {epoch}")

        # Save periodic checkpoints
        if (epoch + 1) % self.config.training.save_interval == 0:
            periodic_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
            torch.save(checkpoint, periodic_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        return checkpoint.get("epoch", 0)

    def save_metrics_json(self):
        """Save training metrics to JSON."""
        metrics_file = self.checkpoint_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
