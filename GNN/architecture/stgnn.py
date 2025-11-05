from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_conv1d(
        x: torch.Tensor,
        conv: nn.Conv1d,
        dilation: int) -> torch.Tensor:
    """
    Apply 1D conv with causal padding on time dimension.
    x: [B*N, C_in, T]
    conv: Conv1d with given dilation and kernel_size
    """
    k = conv.kernel_size[0]
    pad = (k - 1) * dilation
    x = F.pad(x, (pad, 0))  # pad left
    return conv(x)


class TemporalBlock(nn.Module):
    """
    Dilated causal temporal convolution with residual and bottleneck.
    Operates per-node (shared weights across nodes).

    Input:  [B, T, N, C_in]
    Output: [B, T, N, C_out]
    """
    def __init__(self, c_in: int, c_out: int,
                 kernel_size: int, dilation: int,
                 dropout: float):
        super().__init__()
        self.dilation = dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.downsample = (nn.Conv1d(c_in, c_out, kernel_size=1)
                           if c_in != c_out else nn.Identity()
                           )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reshape to [B*N, C, T]
        B, T, N, C = x.shape
        z = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        y = causal_conv1d(z, self.conv1, self.conv1.dilation[0])
        y = self.act(y)
        y = self.dropout(y)
        y = causal_conv1d(y, self.conv2, self.conv2.dilation[0])
        y = self.dropout(y)
        # residual
        res = (causal_conv1d(z, self.downsample, dilation=1)
               if not isinstance(self.downsample, nn.Identity) else z
               )
        y = self.act(y + res)
        # back to [B, T, N, C_out]
        T_out = y.shape[-1]
        y = y.reshape(B, N, -1, T_out).permute(0, 3, 1, 2)
        return y


class GraphConv(nn.Module):
    """
    First-order graph convolution: H = sigma(A_hat X W + b)
    Applied per time step.
    Input:  [B, T, N, C_in]
    Output: [B, T, N, C_out]
    """
    def __init__(self, c_in: int, c_out: int,
                 A_hat: torch.Tensor, bias: bool = True):
        super().__init__()
        # fixed adjacency
        self.A_hat = nn.Parameter(A_hat, requires_grad=False)
        self.lin = nn.Linear(c_in, c_out, bias=bias)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C_in]
        B, T, N, C = x.shape
        # A_hat X: [N, N] @ [B, T, N, C] -> [B, T, N, C]
        x_agg = torch.einsum("ij,btjc->btic", self.A_hat, x)
        y = self.lin(x_agg)
        return self.act(y)


class STBlock(nn.Module):
    """
    Temporal -> Graph -> Temporal block with residual connections.
    """
    def __init__(self, c_in: int, c_hidden: int,
                 c_out: int, A_hat: torch.Tensor,
                 t_kernel: int, dilation: int,
                 dropout: float):
        super().__init__()
        self.temp1 = TemporalBlock(c_in, c_hidden, kernel_size=t_kernel,
                                   dilation=dilation, dropout=dropout)
        self.gconv = GraphConv(c_hidden, c_hidden, A_hat=A_hat)
        self.temp2 = TemporalBlock(c_hidden, c_out, kernel_size=t_kernel,
                                   dilation=dilation, dropout=dropout)
        self.res_proj = (nn.Linear(c_in, c_out)
                         if c_in != c_out else nn.Identity())
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.temp1(x)
        x = self.gconv(x)
        x = self.temp2(x)
        if isinstance(self.res_proj, nn.Identity):
            out = self.act(x + res)
        else:
            # project residual across channel dim
            B, T, N, C_in = res.shape
            r = (self.res_proj(res.reshape(B * T * N, C_in))
                 .reshape(B, T, N, -1))
            out = self.act(x + r)
        return out


class STGNN(nn.Module):
    """
    Spatio-Temporal GNN for multi-horizon forecasting.
    - Input:  [B, L_enc, N, F]
    - Output: [B, H, N, Q]
    """
    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        horizon: int,
        A_hat: torch.Tensor,
        hidden_channels: int = 64,
        num_blocks: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ):
        super().__init__()
        self.horizon = horizon
        self.quantiles = quantiles
        self.A_hat = A_hat

        ch_in = in_features
        ch = hidden_channels
        blocks = []
        for b in range(num_blocks):
            dilation = 2 ** b
            blocks.append(STBlock(
                c_in=ch_in if b == 0 else ch,
                c_hidden=ch,
                c_out=ch,
                A_hat=A_hat,
                t_kernel=kernel_size,
                dilation=dilation,
                dropout=dropout
            ))
        self.blocks = nn.ModuleList(blocks)

        # Decoder: take last time step representation per node and map to H * Q
        self.decoder = nn.Linear(ch, horizon * len(quantiles))

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        # x_hist: [B, L_enc, N, F]
        x = x_hist
        for blk in self.blocks:
            x = blk(x)
        # take last time step
        h_last = x[:, -1, :, :]  # [B, N, C]
        out = self.decoder(h_last)  # [B, N, H*Q]
        B, N, _ = out.shape
        out = (out.view(B, N, self.horizon, len(self.quantiles))
               .permute(0, 2, 1, 3))  # [B, H, N, Q]
        return out


class QuantileLoss(nn.Module):
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.register_buffer("q", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        # y_pred: [B, H, N, Q], y_true: [B, H, N]
        q = self.q.view(1, 1, 1, -1)
        e = y_true.unsqueeze(-1) - y_pred
        loss = torch.maximum(q * e, (q - 1) * e).mean()
        return loss
