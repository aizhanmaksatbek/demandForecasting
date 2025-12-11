import torch
import torch.nn as nn
import torch.nn.functional as F
from TFT.architecture.tft import TemporalFusionTransformer


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """Normalizes the adjacency matrix into the symmetric normalized form.
    Compute {A} = D^{-1/2} (A + I)
    D^{-1/2} for dense A
    """
    N = A.size(0)
    A_hat = A + torch.eye(N, device=A.device, dtype=A.dtype)
    deg = A_hat.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # X: [N, F], A_norm: [N, N]
        return self.lin(A_norm @ X)


class RelationalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int,
                 out_dim: int, dropout: float = 0.1
                 ):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(X, A_norm)
        h = F.relu(h)
        h = self.dropout(h)
        z = self.gcn2(h, A_norm)
        return z  # [N, out_dim]


class HybridGNNTFT(nn.Module):
    """Hybrid model combining a GNN and TFT.
    GNN generates node embeddings to be concatenated to TFT static inputs.
    """

    def __init__(
        self,
        static_input_dims,
        past_input_dims,
        future_input_dims,
        tft_d_model: int = 64,
        tft_hidden_dim: int = 128,
        tft_heads: int = 4,
        tft_lstm_hidden: int = 64,
        tft_lstm_layers: int = 1,
        tft_dropout: float = 0.1,
        tft_num_quantiles: int = 3,
        gnn_in_dim: int = 32,
        gnn_hidden: int = 64,
        gnn_out_dim: int = 32,
        gnn_dropout: float = 0.1,
    ):
        super().__init__()
        # Relational encoder
        self.rel_encoder = RelationalEncoder(
            in_dim=gnn_in_dim,
            hidden_dim=gnn_hidden,
            out_dim=gnn_out_dim,
            dropout=gnn_dropout,
        )
        self.gnn_out_dim = gnn_out_dim

        self.tft = TemporalFusionTransformer(
            static_input_dims=static_input_dims,
            past_input_dims=past_input_dims,
            future_input_dims=future_input_dims,
            d_model=tft_d_model,
            hidden_dim=tft_hidden_dim,
            n_heads=tft_heads,
            lstm_hidden_size=tft_lstm_hidden,
            lstm_layers=tft_lstm_layers,
            dropout=tft_dropout,
            num_quantiles=tft_num_quantiles,
        )

    def forward(
        self,
        past_inputs: torch.Tensor,
        future_inputs: torch.Tensor,
        static_inputs: torch.Tensor,
        node_ids: torch.Tensor,
        A_norm: torch.Tensor,
        X_nodes: torch.Tensor,
        return_attention: bool = False,
    ):
        # Compute node embeddings once per forward and select batch rows
        Z = self.rel_encoder(X_nodes, A_norm)  # [N_nodes, gnn_out_dim]
        z_batch = Z[node_ids]  # [B, gnn_out_dim]
        # Concatenate with static inputs along feature dim
        static_aug = torch.cat([static_inputs, z_batch], dim=1)
        return self.tft(
            past_inputs,
            future_inputs,
            static_aug,
            return_attention=return_attention,
        )
