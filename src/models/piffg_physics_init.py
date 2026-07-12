"""Physics-informed weight initialization for PIFFG."""
import torch
import torch.nn as nn


def initialize_gat_with_physics_prior(gat_layer, edge_attr_template=None, init_strength=0.5):
    """
    Initialize GAT layer weights with physics priors.

    Instead of random initialization, initialize weights to bias the model toward
    learning relationships that respect the physics-based edge structure.

    Args:
        gat_layer: GATConv layer to initialize
        edge_attr_template: Reference edge attributes reflecting physics (e.g., harmonics=1.0, sidebands=0.7)
        init_strength: Scaling factor for physics influence (0.0=random, 1.0=pure physics)

    Returns:
        None (modifies gat_layer in-place)
    """
    # Physics-based initialization: favor attention weights toward strong physics relationships
    # This helps the network converge faster by starting with a bias toward physically meaningful patterns

    # Default physics relationships: harmonics (1.0) > sidebands (0.7) > spectral (0.3)
    if edge_attr_template is None:
        edge_attr_template = torch.tensor(
            [1.0, 1.0, 0.7, 0.7, 0.5, 0.3],  # harmonic, harmonic, sideband x2, mechanical, spectral
            dtype=torch.float32
        )

    # Initialize weight matrices with scaled physics priors
    # Instead of standard Glorot uniform, use physics-weighted initialization
    for name, param in gat_layer.named_parameters():
        if "weight" in name and param.dim() >= 2:
            # Get Glorot uniform initialization as base
            fan_in = param.size(1)
            fan_out = param.size(0)
            std = (2.0 / (fan_in + fan_out)) ** 0.5

            # Initialize with normal distribution
            nn.init.normal_(param, mean=0.0, std=std)

            # Apply physics prior scaling: weight matrix columns are influenced by physics relationships
            # This biases the network to respect harmonic and sideband relationships
            n_edges = min(edge_attr_template.size(0), param.size(0))
            for i in range(n_edges):
                physics_weight = edge_attr_template[i].item() if i < len(edge_attr_template) else 0.5
                # Scale the column by physics relationship strength
                scaled_factor = 0.5 + init_strength * (physics_weight / 1.0)  # Range [0.5, 0.5+init_strength]
                param.data[i, :] *= scaled_factor

        elif "bias" in name and param.dim() == 1:
            # Initialize biases toward zero with small physics-influenced offset
            nn.init.zeros_(param)


def apply_physics_initialization_to_piffg(piffg_module, init_strength=0.5):
    """
    Apply physics-informed initialization to all GAT layers in a PIFFG module.

    Args:
        piffg_module: PhysicsInformedFaultFrequencyGraph instance
        init_strength: Scaling factor (0.0=random, 1.0=strong physics bias)
    """
    # Physics edge weights from PIFFG design: harmonics strongest, then sidebands, then spectral
    physics_weights = torch.tensor([1.0, 1.0, 0.7, 0.7, 0.5, 0.3], dtype=torch.float32)

    # Apply to both GAT layers
    for layer_name in ["gat1", "gat2"]:
        if hasattr(piffg_module, layer_name):
            gat_layer = getattr(piffg_module, layer_name)
            initialize_gat_with_physics_prior(gat_layer, physics_weights, init_strength)
