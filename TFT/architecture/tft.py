"""
Temporal Fusion Transformer (TFT) — PyTorch single-file implementation

Core features:
- Variable Selection Networks (static, encoder-time, decoder-time)
- Gated Residual Networks with GLU and LayerNorm
- LSTM encoder-decoder backbone with static-conditioned initial state
- Multi-head attention with causal masking
- Position-wise GRN and quantile output head
- Returns interpretable weights (variable selection, attention)

See: Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (2021)
"""

from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_features(x: torch.Tensor, dims: List[int]) -> List[torch.Tensor]:
    assert x.size(-1) == sum(dims), f"Expected last dim {sum(dims)}, got {x.size(-1)}"
    outs = []
    start = 0
    for d in dims:
        outs.append(x[..., start:start + d])
        start += d
    return outs


def _ensure_list(
    x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    dims: List[int],
) -> List[torch.Tensor]:
    if isinstance(x, (list, tuple)):
        assert len(x) == len(dims), f"Expected {len(dims)} vars, got {len(x)}"
        return list(x)
    else:
        return _split_features(x, dims)


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int, d_out: int, context_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * d_out)
        self.elu = nn.ELU()
        self.layer_norm = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.context_proj = nn.Linear(context_dim, d_in) if context_dim is not None else None
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _glu(x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * torch.sigmoid(b)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if (context is not None) and (self.context_proj is not None):
            if context.dim() == x.dim() - 1:
                for _ in range(x.dim() - context.dim()):
                    context = context.unsqueeze(1)
            x_in = x + self.context_proj(context)
        else:
            x_in = x
        h = self.elu(self.fc1(x_in))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self._glu(h)
        y = self.layer_norm(self.skip(x) + h)
        return y


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dims: List[int], d_model: int, hidden_dim: int, context_dim: Optional[int], temporal: bool, dropout: float = 0.0):
        super().__init__()
        self.input_dims = input_dims
        self.V = len(input_dims)
        self.temporal = temporal
        self.value_encoders = nn.ModuleList([nn.Linear(d, d_model) for d in input_dims])
        self.weight_net = GatedResidualNetwork(sum(input_dims), hidden_dim, self.V, context_dim=context_dim, dropout=dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: Union[List[torch.Tensor], torch.Tensor], context: Optional[torch.Tensor] = None):
        xs = inputs if isinstance(inputs, list) else _split_features(inputs, self.input_dims)
        assert len(xs) == self.V
        if self.temporal:
            encoded = [enc(x) for enc, x in zip(self.value_encoders, xs)]
            enc_stack = torch.stack(encoded, dim=-2)  # [B, T, V, d_model]
            x_cat = torch.cat(xs, dim=-1)            # [B, T, sum(d)]
            weights = self.softmax(self.weight_net(x_cat, context=context))  # [B, T, V]
            selected = torch.sum(enc_stack * weights.unsqueeze(-1), dim=-2)  # [B, T, d_model]
            return selected, weights
        else:
            encoded = [enc(x) for enc, x in zip(self.value_encoders, xs)]
            enc_stack = torch.stack(encoded, dim=-2)  # [B, V, d_model]
            x_cat = torch.cat(xs, dim=-1)            # [B, sum(d)]
            weights = self.softmax(self.weight_net(x_cat, context=context))  # [B, V]
            selected = torch.sum(enc_stack * weights.unsqueeze(-1), dim=-2)  # [B, d_model]
            return selected, weights


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        static_input_dims: List[int],
        past_input_dims: List[int],
        future_input_dims: List[int],
        d_model: int = 64,
        hidden_dim: int = 128,
        n_heads: int = 4,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        num_quantiles: int = 3,
    ):
        super().__init__()
        self.static_input_dims = static_input_dims
        self.past_input_dims = past_input_dims
        self.future_input_dims = future_input_dims
        self.V_enc = len(past_input_dims)
        self.V_dec = len(future_input_dims)
        self.V_stat = len(static_input_dims)
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_quantiles = num_quantiles
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        self.static_vsn = VariableSelectionNetwork(static_input_dims, d_model, hidden_dim, context_dim=None, temporal=False, dropout=dropout)
        self.encoder_vsn = VariableSelectionNetwork(past_input_dims, d_model, hidden_dim, context_dim=d_model, temporal=True, dropout=dropout)
        self.decoder_vsn = VariableSelectionNetwork(future_input_dims, d_model, hidden_dim, context_dim=d_model, temporal=True, dropout=dropout)

        self.static_context_selection = GatedResidualNetwork(d_model, hidden_dim, d_model, context_dim=None, dropout=dropout)
        self.static_context_enrichment = GatedResidualNetwork(d_model, hidden_dim, d_model, context_dim=None, dropout=dropout)
        self.static_context_state_h = GatedResidualNetwork(d_model, hidden_dim, d_model, context_dim=None, dropout=dropout)
        self.static_context_state_c = GatedResidualNetwork(d_model, hidden_dim, d_model, context_dim=None, dropout=dropout)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
        self.lstm_proj = nn.Linear(lstm_hidden_size, d_model)

        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.attn_skip_gating = GatedResidualNetwork(d_model, hidden_dim, d_model, context_dim=None, dropout=dropout)
        self.positionwise_grn = GatedResidualNetwork(d_model, hidden_dim, d_model, context_dim=d_model, dropout=dropout)

        self.output_proj = nn.Linear(d_model, num_quantiles)

        self.h_init_proj = nn.Linear(d_model, lstm_hidden_size)
        self.c_init_proj = nn.Linear(d_model, lstm_hidden_size)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _causal_mask(size_q: int, size_k: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size_q, size_k, device=device, dtype=torch.bool), diagonal=1)

    def _init_lstm_state(self, static_ctx_h: torch.Tensor, static_ctx_c: torch.Tensor, batch_size: int):
        h0_single = self.h_init_proj(static_ctx_h)  # [B, H]
        c0_single = self.c_init_proj(static_ctx_c)
        h0 = torch.stack([h0_single] * self.lstm_layers, dim=0).contiguous()
        c0 = torch.stack([c0_single] * self.lstm_layers, dim=0).contiguous()
        return h0, c0

    def forward(
        self,
        past_inputs: Union[List[torch.Tensor], torch.Tensor],
        future_inputs: Union[List[torch.Tensor], torch.Tensor],
        static_inputs: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B = (past_inputs[0].size(0) if isinstance(past_inputs, list) else past_inputs.size(0))
        device = (past_inputs[0].device if isinstance(past_inputs, list) else past_inputs.device)

        past_list = _ensure_list(past_inputs, self.past_input_dims)
        future_list = _ensure_list(future_inputs, self.future_input_dims)
        if static_inputs is None:
            static_inputs = torch.zeros(B, sum(self.static_input_dims), device=device)
        static_list = _ensure_list(static_inputs, self.static_input_dims)

        L_enc = past_list[0].size(1)
        L_dec = future_list[0].size(1)
        L_total = L_enc + L_dec

        s_selected, s_weights = self.static_vsn(static_list, context=None)  # [B, d_model], [B, V_static]
        c_selection = self.static_context_selection(s_selected)
        c_enrichment = self.static_context_enrichment(s_selected)
        c_state_h = self.static_context_state_h(s_selected)
        c_state_c = self.static_context_state_c(s_selected)

        enc_selected, enc_w = self.encoder_vsn(past_list, context=c_selection)   # [B, L_enc, d_model], [B, L_enc, V_enc]
        dec_selected, dec_w = self.decoder_vsn(future_list, context=c_selection) # [B, L_dec, d_model], [B, L_dec, V_dec]

        seq_inputs = torch.cat([enc_selected, dec_selected], dim=1)  # [B, L_total, d_model]
        h0, c0 = self._init_lstm_state(c_state_h, c_state_c, B)
        lstm_out, _ = self.lstm(seq_inputs, (h0, c0))                # [B, L_total, H]
        lstm_out = self.lstm_proj(lstm_out)                          # [B, L_total, d_model]

        q = lstm_out[:, L_enc:, :]
        k = lstm_out
        v = lstm_out
        attn_mask = self._causal_mask(L_dec, L_total, device=device)
        attn_out, attn_weights = self.mha(q, k, v, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)  # [B, n_heads, L_dec, L_total]

        dec_lstm = lstm_out[:, L_enc:, :]
        attn_skip = self.attn_skip_gating(dec_lstm)
        attn_residual = F.layer_norm(dec_lstm + attn_out, normalized_shape=(dec_lstm.size(-1),))

        dec_enriched = self.positionwise_grn(attn_residual, context=c_enrichment)

        preds = self.output_proj(self.dropout(dec_enriched))  # [B, L_dec, Q]

        out = {
            "prediction": preds,
            "encoder_variable_importance": enc_w,
            "decoder_variable_importance": dec_w,
            "static_variable_importance": s_weights,
        }
        if return_attention:
            out["attn_weights"] = attn_weights
        return out


class QuantileLoss(nn.Module):
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        q = self.quantiles.view(1, 1, -1)  # [1,1,Q]
        e = y_true.unsqueeze(-1) - y_pred  # [B,T,Q]
        loss = torch.maximum(q * e, (q - 1) * e)
        return loss.mean()
