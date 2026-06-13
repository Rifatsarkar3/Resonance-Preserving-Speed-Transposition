"""Configuration loader for WaPIGT."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any
import yaml


@dataclass
class DataConfig:
    cwru_raw_root: str
    pu_raw_root: str
    jnu_raw_root: str
    processed_root: str = "data/processed"
    splits_root: str = "data/splits"


@dataclass
class ModelConfig:
    variant: str = "3M"
    hidden_dim: int = 96
    n_encoder_layers: int = 4
    n_heads: int = 8
    mlp_dim: int = 384
    dropout: float = 0.4
    n_gat_heads: int = 4
    gat_dropout: float = 0.4
    filter_order: int = 8
    n_frames: int = 16
    scr_lambda: float = 0.1
    scr_warmup_epochs: int = 10
    triplet_lambda: float = 0.1
    triplet_margin: float = 0.5
    triplet_warmup_epochs: int = 20

    def __post_init__(self):
        """Resolve hidden_dim based on variant if using defaults."""
        if self.variant == "3M" and self.hidden_dim == 96:
            self.hidden_dim = 96
        elif self.variant == "5M":
            self.hidden_dim = 128


@dataclass
class TrainingConfig:
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-3
    batch_size: int = 64
    n_epochs: int = 200
    gradient_clip_norm: float = 1.0
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    precision: str = "bf16"
    checkpoint_dir: str = "outputs/checkpoints"
    save_interval: int = 10
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Dataset-specific overrides (Phase 4: regularization for small-data JNU)
    jnu_dropout_override: float = 0.55
    jnu_weight_decay_override: float = 5.0e-3
    jnu_warmup_epochs_override: int = 5


@dataclass
class EvaluationConfig:
    metrics_dir: str = "outputs/metrics"
    latency_n_warmup: int = 50
    latency_n_runs: int = 500
    log_dir: str = "outputs/logs"


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        data = DataConfig(**config_dict.get('data', {}))
        model = ModelConfig(**config_dict.get('model', {}))
        training = TrainingConfig(**config_dict.get('training', {}))
        evaluation = EvaluationConfig(**config_dict.get('evaluation', {}))

        return cls(
            data=data,
            model=model,
            training=training,
            evaluation=evaluation,
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'cuda'),
        )

    def validate_paths(self):
        """Validate that all dataset paths exist."""
        paths = [self.data.cwru_raw_root, self.data.pu_raw_root, self.data.jnu_raw_root]
        for path_str in paths:
            if path_str.startswith("E:/"):
                # Skip validation for example paths
                continue
            path = Path(path_str)
            if not path.exists():
                print(f"WARNING: Dataset path not found: {path}")
                print(f"Update config.yaml before running data preprocessing")
