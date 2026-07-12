"""Physics-Informed Fault-Frequency Graph (PIFFG)."""
import torch
import torch.nn as nn
import math
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from src.utils.fault_frequencies import compute_fault_frequencies


class PhysicsInformedFaultFrequencyGraph(nn.Module):
    """
    Physics-informed GAT operating on canonical bearing fault frequencies.

    Nodes: {BPFO, BPFI, FTF, BSF, f_s, 2f_s, 3f_s}
    Edges: Harmonic and sideband relationships derived from bearing mechanics.
    """

    def __init__(self, hidden_dim: int = 96, n_gat_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_gat_heads = n_gat_heads
        self.n_nodes = 7

        # Node features: [freq_normalized, sin(2π*freq/f_nyquist), cos(2π*freq/f_nyquist)]
        self.gat1 = GATConv(3, hidden_dim // 2, heads=n_gat_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv((hidden_dim // 2) * n_gat_heads, hidden_dim, heads=1, concat=False)

    def _build_graph(self, bearing_params_list, fs_sampling):
        """Build fault-frequency graph for a batch."""
        graphs = []

        for bearing_params in bearing_params_list:
            # Compute fault frequencies
            freqs = compute_fault_frequencies(
                N_balls=bearing_params.get("N_balls", 9),
                d_mm=bearing_params.get("d_mm", 7.94),
                D_mm=bearing_params.get("D_mm", 39.04),
                alpha_deg=bearing_params.get("alpha_deg", 0.0),
                f_shaft_hz=bearing_params.get("f_s", 29.95),
            )

            # Use bearing orders (fault_freq / shaft_freq) as node features.
            # Orders are speed-invariant: BPFO_order = BPFO/f_shaft = N/2*(1-(d/D)*cosα)
            # regardless of RPM. This gives the GAT meaningful, distinct features for
            # each fault type at any operating speed, solving the freq/nyquist≈0 issue
            # for high-fs datasets (JNU 50kHz, PU 64kHz).
            # Shaft-harmonic nodes get orders 1, 2, 3 by definition.
            # Normalize by 6.0 (max expected bearing order) to put features in [0, 1].
            f_shaft = bearing_params.get("f_s", 29.95)
            ORDER_NORM = 6.0  # normalizer: covers all typical bearing orders up to ~6x shaft

            node_freqs = [
                freqs["BPFO"],
                freqs["BPFI"],
                freqs["FTF"],
                freqs["BSF"],
                freqs["f_s"],
                freqs["2f_s"],
                freqs["3f_s"],
            ]

            x = []
            for freq in node_freqs:
                order = (freq / (f_shaft + 1e-8)) / ORDER_NORM  # normalized order [0, 1]
                order = min(order, 1.0)
                sin_feat = math.sin(2 * math.pi * order)
                cos_feat = math.cos(2 * math.pi * order)
                x.append([order, sin_feat, cos_feat])

            x = torch.tensor(x, dtype=torch.float32)

            # Physics-based edges (directed, will be made bidirectional)
            edge_list = [
                (4, 5, 1.0),   # f_s → 2f_s (harmonic)
                (4, 6, 1.0),   # f_s → 3f_s (harmonic)
                (4, 0, 0.7),   # f_s → BPFO (sideband)
                (4, 1, 0.7),   # f_s → BPFI (sideband)
                (2, 3, 0.5),   # FTF → BSF (mechanical)
                (0, 1, 0.3),   # BPFO ↔ BPFI (spectral)
            ]

            # Make bidirectional
            edge_index = []
            edge_attr = []
            for src, dst, weight in edge_list:
                edge_index.append([src, dst])
                edge_index.append([dst, src])
                edge_attr.append(weight)
                edge_attr.append(weight)

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            graphs.append(graph)

        return Batch.from_data_list(graphs)

    def forward(self, bearing_params_list, fs_sampling: float = 64000.0):
        """
        Forward pass.

        Args:
            bearing_params_list: List[Dict] with bearing geometry and shaft frequency
            fs_sampling: Sampling frequency in Hz

        Returns:
            graph_embedding: (B, hidden_dim) graph-level embeddings
        """
        # Build batch graph
        batch = self._build_graph(bearing_params_list, fs_sampling)
        batch = batch.to(next(self.parameters()).device)

        # GAT layers
        x = batch.x
        x = self.gat1(x, batch.edge_index)
        x = torch.relu(x)
        x = self.gat2(x, batch.edge_index)

        # Global mean pooling over nodes
        batch_vec = batch.batch
        graph_embedding = global_mean_pool(x, batch_vec)

        return graph_embedding
