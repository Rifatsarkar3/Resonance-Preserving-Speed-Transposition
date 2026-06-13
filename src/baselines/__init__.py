"""Baseline models for comparison with WaPIGT."""

from src.baselines.wdcnn import WDCNN
from src.baselines.ticnn import TICNN
from src.baselines.mscnn import MSCNN
from src.baselines.physformer import PhysFormer
from src.baselines.vit1d import ViT1D
from src.baselines.gnnfault import GNNFault
from src.baselines.checkpoint_manager import CheckpointManager
from src.baselines.baseline_orchestrator import BaselineTrainer

__all__ = [
    "WDCNN",
    "TICNN",
    "MSCNN",
    "PhysFormer",
    "ViT1D",
    "GNNFault",
    "CheckpointManager",
    "BaselineTrainer",
]

BASELINE_REGISTRY = {
    "WDCNN": WDCNN,
    "TICNN": TICNN,
    "MSCNN": MSCNN,
    "PhysFormer": PhysFormer,
    "ViT1D": ViT1D,
    "GNNFault": GNNFault,
}
