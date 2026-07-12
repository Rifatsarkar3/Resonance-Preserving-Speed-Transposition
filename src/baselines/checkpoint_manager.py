"""Checkpoint and recovery management for baseline training."""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages baseline training checkpoints and recovery state."""

    def __init__(self, checkpoint_root: str = "outputs/baselines"):
        """Initialize checkpoint manager."""
        self.checkpoint_root = Path(checkpoint_root)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)

        # Master tracking file
        self.tracking_file = self.checkpoint_root / "training_manifest.json"
        self.tracking_data = self._load_tracking()

    def _load_tracking(self) -> Dict[str, Any]:
        """Load or initialize tracking file."""
        if self.tracking_file.exists():
            with open(self.tracking_file) as f:
                return json.load(f)
        return {
            "created": datetime.now().isoformat(),
            "baselines": {},
            "total_completed_trials": 0,
            "failed_trials": [],
        }

    def _save_tracking(self):
        """Save tracking file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)

    def get_baseline_dir(self, baseline_name: str) -> Path:
        """Get checkpoint directory for a baseline."""
        baseline_dir = self.checkpoint_root / baseline_name
        baseline_dir.mkdir(parents=True, exist_ok=True)
        return baseline_dir

    def get_trial_checkpoint_dir(self, baseline_name: str, trial_id: int) -> Path:
        """Get checkpoint directory for a specific trial."""
        trial_dir = self.get_baseline_dir(baseline_name) / f"trial_{trial_id:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        return trial_dir

    def get_trial_checkpoint_path(self, baseline_name: str, trial_id: int) -> Path:
        """Get path to best checkpoint for a trial."""
        trial_dir = self.get_trial_checkpoint_dir(baseline_name, trial_id)
        return trial_dir / "best.pt"

    def get_trial_metrics_path(self, baseline_name: str, trial_id: int) -> Path:
        """Get path to metrics JSON for a trial."""
        trial_dir = self.get_trial_checkpoint_dir(baseline_name, trial_id)
        return trial_dir / "metrics.json"

    def get_baseline_results_path(self, baseline_name: str) -> Path:
        """Get path to aggregated results for a baseline."""
        return self.get_baseline_dir(baseline_name) / "results.json"

    def get_baseline_log_path(self, baseline_name: str) -> Path:
        """Get path to baseline training log."""
        return self.get_baseline_dir(baseline_name) / "training.log"

    def is_trial_completed(self, baseline_name: str, trial_id: int) -> bool:
        """Check if a trial has been completed."""
        if baseline_name not in self.tracking_data["baselines"]:
            return False

        baseline_data = self.tracking_data["baselines"][baseline_name]
        return trial_id in baseline_data.get("completed_trials", [])

    def mark_trial_completed(self, baseline_name: str, trial_id: int,
                            results: Dict[str, Any]):
        """Mark a trial as completed."""
        if baseline_name not in self.tracking_data["baselines"]:
            self.tracking_data["baselines"][baseline_name] = {
                "completed_trials": [],
                "failed_trials": [],
                "last_updated": None,
                "results": []
            }

        baseline_data = self.tracking_data["baselines"][baseline_name]

        if trial_id not in baseline_data["completed_trials"]:
            baseline_data["completed_trials"].append(trial_id)
            baseline_data["completed_trials"].sort()
            baseline_data["last_updated"] = datetime.now().isoformat()
            baseline_data["results"].append(results)
            self.tracking_data["total_completed_trials"] += 1

        self._save_tracking()
        logger.info(f"Marked {baseline_name} trial {trial_id} as completed")

    def mark_trial_failed(self, baseline_name: str, trial_id: int,
                         error: str):
        """Mark a trial as failed."""
        if baseline_name not in self.tracking_data["baselines"]:
            self.tracking_data["baselines"][baseline_name] = {
                "completed_trials": [],
                "failed_trials": [],
                "last_updated": None,
            }

        baseline_data = self.tracking_data["baselines"][baseline_name]

        if trial_id not in baseline_data["failed_trials"]:
            baseline_data["failed_trials"].append({
                "trial_id": trial_id,
                "error": error,
                "timestamp": datetime.now().isoformat()
            })
            baseline_data["last_updated"] = datetime.now().isoformat()

        self._save_tracking()
        self.tracking_data["failed_trials"].append({
            "baseline": baseline_name,
            "trial_id": trial_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        self._save_tracking()
        logger.warning(f"Marked {baseline_name} trial {trial_id} as failed: {error}")

    def get_completed_trials(self, baseline_name: str) -> list:
        """Get list of completed trial IDs for a baseline."""
        if baseline_name not in self.tracking_data["baselines"]:
            return []
        return self.tracking_data["baselines"][baseline_name].get("completed_trials", [])

    def get_baseline_progress(self, baseline_name: str, total_trials: int = 50) -> Dict[str, Any]:
        """Get progress summary for a baseline."""
        completed = len(self.get_completed_trials(baseline_name))
        failed = len(self.tracking_data["baselines"].get(baseline_name, {}).get("failed_trials", []))
        remaining = total_trials - completed - failed

        return {
            "baseline": baseline_name,
            "completed": completed,
            "failed": failed,
            "remaining": remaining,
            "total": total_trials,
            "percent_complete": 100 * completed / total_trials if total_trials > 0 else 0,
        }

    def get_all_progress(self, total_trials_per_baseline: int = 50) -> Dict[str, Any]:
        """Get progress for all baselines."""
        progress = {}
        for baseline_name in self.tracking_data["baselines"].keys():
            progress[baseline_name] = self.get_baseline_progress(
                baseline_name, total_trials_per_baseline
            )

        total_completed = sum(p["completed"] for p in progress.values())
        total_possible = len(progress) * total_trials_per_baseline

        return {
            "by_baseline": progress,
            "total_completed": total_completed,
            "total_possible": total_possible,
            "percent_complete": 100 * total_completed / total_possible if total_possible > 0 else 0,
            "failed_trials": self.tracking_data["failed_trials"],
        }

    def save_trial_metrics(self, baseline_name: str, trial_id: int,
                          metrics: Dict[str, Any]):
        """Save metrics for a trial."""
        metrics_path = self.get_trial_metrics_path(baseline_name, trial_id)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def load_trial_metrics(self, baseline_name: str, trial_id: int) -> Optional[Dict[str, Any]]:
        """Load metrics for a trial."""
        metrics_path = self.get_trial_metrics_path(baseline_name, trial_id)
        if not metrics_path.exists():
            return None
        with open(metrics_path) as f:
            return json.load(f)

    def get_recovery_status(self) -> Dict[str, Any]:
        """Get overall recovery status."""
        return {
            "manifest_file": str(self.tracking_file),
            "manifest_exists": self.tracking_file.exists(),
            "total_completed": self.tracking_data["total_completed_trials"],
            "failed_trials": len(self.tracking_data["failed_trials"]),
            "baselines_started": len(self.tracking_data["baselines"]),
        }


if __name__ == "__main__":
    # Test checkpoint manager
    logging.basicConfig(level=logging.INFO)

    cm = CheckpointManager()
    print("Recovery Status:")
    print(json.dumps(cm.get_recovery_status(), indent=2))
    print("\nProgress Summary:")
    print(json.dumps(cm.get_all_progress(), indent=2))
