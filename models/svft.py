import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from typing import Literal


class SVFTLayer(nn.Module):
    def __init__(
        self,
        u: torch.Tensor,
        s: torch.Tensor,
        v: torch.Tensor,
        off_diag: int,
        pattern: str = "banded",
        rank: int = None,
        fill_orthonormal: bool = False,
    ):
        """
        @inputs:
            u: torch.Tensor. Left singular vectors of pre-trained weight matrix
            s: torch.Tensor. Singular values of pre-trained weight matrix
            v: torch.Tensor. Right singular vectors of pre-trained weight matrix
            off_diag: int. Total off-diagonals to be used to populate matrix M (as referred in main paper)
            pattern: str. Choices: "banded", "random", "top_k". Using "banded" with off_diag=1 simulates SVFT-plain
            rank: int. Constraints how many singular vectors and values to use.
            fill_orthonormal: bool. To determine if random orthonormal basis should be used
        """

        super().__init__()

        self.off_diag = off_diag
        rank = s.shape[0] if rank is None else min(s.shape[0], rank)
        self.n = rank
        diff_rank = s.shape[0] - rank

        if fill_orthonormal:
            Q_u = torch.randn_like(u).to(s.device)
            torch.nn.init.orthogonal_(Q_u)
            Q_v = torch.randn_like(v).to(s.device)
            torch.nn.init.orthogonal_(Q_v)

            u = torch.cat([u[:, :rank], Q_u[:, :diff_rank]], dim=1)
            v = torch.cat([v[:rank, :], Q_v[:diff_rank, :]], dim=0)
            s = torch.cat([s[:rank], torch.zeros(diff_rank).to(s.device)], dim=0)
            self.n = s.shape[0]

        else:
            s = s[:rank]
            u = u[:, :rank]
            v = v[:rank, :]

        self.u = nn.Parameter(u.clone().detach().contiguous(), requires_grad=False)

        s_pre = s.cpu().detach().clone().contiguous()
        self.s_pre_edge_index = (
            torch.sparse.spdiags(s_pre, torch.LongTensor([0]), (self.n, self.n))
            .coalesce()
            .indices()
        )
        self.s_pre = nn.Parameter(s_pre, requires_grad=False)

        if pattern == "banded":
            diags = 2 * self.off_diag + 1
            offsets_positive = torch.arange(0, self.off_diag + 1)
            offsets_negative = torch.arange(-1, -self.off_diag - 1, -1)
            self.offsets = torch.cat([offsets_positive, offsets_negative])
            self.s_edge_index = (
                torch.sparse.spdiags(
                    torch.randn([diags, self.n]), self.offsets, (self.n, self.n)
                )
                .coalesce()
                .indices()
            )
            self.s = torch.nn.Parameter(
                torch.zeros(self.s_edge_index.shape[1]), requires_grad=True
            )

        elif pattern == "random":
            print("Random pattern")
            k = self.n * (2 * self.off_diag + 1) - self.off_diag * (self.off_diag + 1)
            rows = torch.randint(0, self.n, (k,))
            cols = torch.randint(0, self.n, (k,))
            self.s_edge_index = torch.stack([rows, cols])
            self.s = torch.nn.Parameter(torch.zeros(k), requires_grad=True)

        elif pattern == "top_k":

            if u.shape == v.shape:
                coeffs = u @ v.T
            else:
                coeffs = u if u.shape[0] == u.shape[1] else v

            k = self.n * (2 * self.off_diag + 1) - self.off_diag * (self.off_diag + 1)
            # Flatten the tensor to 1D
            flattened_tensor = coeffs.contiguous().view(-1)
            _, top_indices_flat = torch.topk(flattened_tensor, k)
            num_rows, num_cols = coeffs.size()
            rows = top_indices_flat // num_cols
            cols = top_indices_flat % num_cols
            self.s_edge_index = torch.stack([rows, cols])
            self.s = torch.nn.Parameter(torch.zeros(k), requires_grad=True)

        torch.nn.init.kaiming_normal_(self.s[None, :])
        self.s.squeeze()

        self.register_buffer("s_pre_row", self.s_pre_edge_index[0])
        self.register_buffer("s_pre_col", self.s_pre_edge_index[1])
        self.register_buffer("s_row", self.s_edge_index[0])
        self.register_buffer("s_col", self.s_edge_index[1])

        self.gate = nn.Parameter(
            torch.tensor([0.0], dtype=torch.float32), requires_grad=True
        )

        self.v = nn.Parameter(v.clone().detach().contiguous(), requires_grad=False)

    def forward(self, x):
        x = x @ self.get_weights()
        return x

    def get_weights(self) -> torch.Tensor:
        s = SparseTensor(
            row=self.s_row, col=self.s_col, value=self.s * F.sigmoid(self.gate)
        )
        s_pre = SparseTensor(row=self.s_pre_row, col=self.s_pre_col, value=self.s_pre)
        del_s = s_pre + s
        weight = (del_s @ self.v).T
        weight = weight @ self.u.T
        return weight

    def merge_and_unload(self):
        return self.get_weights().T.contiguous()


class Linear(nn.Module):
    """
    A drop-in replacement for nn.Linear with extra support for SVFT (Singular Value Factorization Transformer).
    h = W0 x + ∆W x = U (Σ + M) V^T x

    Parameters
    ----------
    nn : _type_
        _description_
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self.register_buffer("_decomposed", torch.tensor(False))

    def lazy_init(
        self,
        mask_pattern: Literal["banded", "random", "top_k"] = "banded",
        off_diag: int = 1,
        rank: int = None,
        fill_orthonormal: bool = False,
    ):
        """Activate SVFT with SVD decomposition.
        Replaces $W$ with $U (Σ + M) V^T$ in `forward()` and sets `_decomposed` flag to True.
        """
        # if already decomposed, do nothing
        if self._decomposed.item():
            return

        # run SVD on weight matrix
        u, s, v = torch.linalg.svd(self.weight.data, full_matrices=False)
        self.svft_layer = SVFTLayer(
            u=u,
            s=s,
            v=v,
            off_diag=off_diag,
            pattern=mask_pattern,
            rank=rank,
            fill_orthonormal=fill_orthonormal,
        )
        self._decomposed.fill_(True)
        self.fix_grad()

    def fix_grad(self, bias_grad: bool = False, *_):
        """Fix the gradient of the weight matrix to False."""
        if self._decomposed.item():
            self.weight.requires_grad_(False)
            self.bias.requires_grad_(bias_grad)

            self.svft_layer.u.requires_grad_(False)
            self.svft_layer.v.requires_grad_(False)
            self.svft_layer.s_pre.requires_grad_(False)

            # only M is trainable
            self.svft_layer.s.requires_grad_(True)
        else:
            raise RuntimeError("SVFT not initialized. Call lazy_init_svft() first.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # treat as nn.Linear
        if not self._decomposed.item():
            return nn.functional.linear(x, self.weight, bias=self.bias)

        # otherwise, use SVFT, which is U (Σ + M) V^T x
        if self.bias is None:
            return self.svft_layer(x)
        return self.svft_layer(x) + self.bias

    def merge_and_unload(self):
        return self.svft_layer.merge_and_unload()
