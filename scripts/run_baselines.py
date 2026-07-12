"""Master script for baseline training with Optuna tuning and checkpoint recovery."""
import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.reproducibility import set_all_seeds
from src.baselines import (
    BASELINE_REGISTRY,
    CheckpointManager,
    BaselineTrainer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


BASELINE_CONFIGS = {
    "WDCNN": {
        "n_trials": 50,
        "description": "1D-CNN gold standard baseline",
    },
    "TICNN": {
        "n_trials": 50,
        "description": "Transfer Inception CNN",
    },
    "MSCNN": {
        "n_trials": 50,
        "description": "Multi-Scale CNN",
    },
    "PhysFormer": {
        "n_trials": 50,
        "description": "Physics-Informed Transformer",
    },
    "ViT1D": {
        "n_trials": 50,
        "description": "Vision Transformer for 1D signals",
    },
    "GNNFault": {
        "n_trials": 50,
        "description": "Graph Neural Network",
    },
}

EVALUATION_TASKS = {
    "PU": "PU_T01",
    "CWRU": "CWRU_T01",
    "JNU": "JNU_T02",
}


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    config = Config.from_yaml(config_path)
    config.validate_paths()
    return config


def print_recovery_status(checkpoint_manager):
    """Print current recovery status."""
    status = checkpoint_manager.get_recovery_status()
    progress = checkpoint_manager.get_all_progress()

    logger.info("\n" + "="*70)
    logger.info("RECOVERY STATUS")
    logger.info("="*70)

    logger.info(f"Manifest file: {status['manifest_file']}")
    logger.info(f"Total completed trials: {status['total_completed']}")
    logger.info(f"Failed trials: {status['failed_trials']}")
    logger.info(f"Baselines started: {status['baselines_started']}")

    logger.info("\n" + "-"*70)
    logger.info("PROGRESS BY BASELINE")
    logger.info("-"*70)

    for baseline_name, baseline_progress in progress["by_baseline"].items():
        logger.info(
            f"{baseline_name:15s}: "
            f"{baseline_progress['completed']:3d}/{baseline_progress['total']:3d} "
            f"({baseline_progress['percent_complete']:5.1f}%) "
            f"| Failed: {baseline_progress['failed']}"
        )

    logger.info("-"*70)
    logger.info(f"Overall: {progress['total_completed']}/{progress['total_possible']} "
                f"({progress['percent_complete']:.1f}%)")
    logger.info("="*70 + "\n")


def show_progress_summary(checkpoint_manager):
    """Show detailed progress summary."""
    print_recovery_status(checkpoint_manager)


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline models with Optuna tuning and checkpoint recovery"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=list(BASELINE_REGISTRY.keys()),
        help="Baselines to train (default: all)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show recovery status and exit",
    )

    args = parser.parse_args()

    checkpoint_manager = CheckpointManager(checkpoint_root="outputs/baselines")

    if args.status:
        show_progress_summary(checkpoint_manager)
        return

    logger.info(f"Starting baseline training at {datetime.now()}")
    print_recovery_status(checkpoint_manager)

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # Train selected baselines
    baselines_to_train = args.baselines
    logger.info(f"Training baselines: {baselines_to_train}")

    for baseline_name in baselines_to_train:
        if baseline_name not in BASELINE_REGISTRY:
            logger.error(f"Unknown baseline: {baseline_name}")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"Training {baseline_name} ({BASELINE_CONFIGS[baseline_name]['description']})")
        logger.info(f"{'='*70}\n")

        model_class = BASELINE_REGISTRY[baseline_name]
        trainer = BaselineTrainer(
            baseline_name=baseline_name,
            model_class=model_class,
            config=config,
            checkpoint_manager=checkpoint_manager,
            n_trials=BASELINE_CONFIGS[baseline_name]["n_trials"],
        )

        # Run all trials with resume support
        results = trainer.run_all_trials(
            dataset=list(EVALUATION_TASKS.keys())[0],
            task=list(EVALUATION_TASKS.values())[0],
        )

        logger.info(f"\n{baseline_name} Training Complete")
        logger.info(f"Total trials: {len(results.get('trial_results', []))}")

    logger.info(f"\n{'='*70}")
    logger.info("Baseline Training Complete!")
    logger.info(f"{'='*70}")
    print_recovery_status(checkpoint_manager)


if __name__ == "__main__":
    main()
