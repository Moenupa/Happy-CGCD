import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    A drop-in replacement for nn.Linear with extra support for LoRA (Low-Rank Adaptation).
    This layer can be used in both training and evaluation modes, with or without LoRA.
    It supports an optional bias term and allows for LoRA to be enabled or disabled.

    Parameters
    ----------
    lora_rank : int, optional
        The rank of the LoRA decomposition. If > 0, LoRA is enabled.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = None,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        # default to require_grad=True, as in nn.Linear
        # lora ver will set this to false in init_lora
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self.init_lora(in_features, out_features, lora_rank, lora_alpha, lora_dropout)

    def init_lora(
        self,
        in_features: int,
        out_features: int,
        r: int,
        alpha: float = None,
        dropout: float = 0.0,
    ):
        """
        Initialize LoRA parameters.
        This method can be called to reinitialize LoRA parameters if needed.
        """
        self.lora_rank = r
        if r > 0:
            self.scaling = 1.0 if alpha is None else alpha / r

            # default grad=True for LoRA BA
            self.lora_a = nn.Parameter(torch.empty((r, in_features)))
            self.lora_b = nn.Parameter(torch.empty((out_features, r)))
            self.lora_dropout = nn.Dropout(dropout)
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
                nn.init.zeros_(self.lora_b)
        else:
            self.scaling = None
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)
            self.lora_dropout = nn.Identity()
        self.fix_grad()

    def forward(self, x: torch.Tensor):
        result = nn.functional.linear(x, self.weight, bias=self.bias)
        if self.lora_rank > 0:
            result += (
                self.lora_dropout(x) @ self.lora_a.T @ self.lora_b.T
            ) * self.scaling
        return result

    def fix_grad(self, bias_grad: bool = False, *_):
        """Fix the gradient of the weight matrix to False."""
        if self.lora_rank > 0:
            self.weight.requires_grad_(False)
            if self.bias is not None:
                self.bias.requires_grad_(bias_grad)
            self.lora_a.requires_grad_(True)
            self.lora_b.requires_grad_(True)
        else:
            # raise RuntimeError("LoRA not initialized. Call init_lora() first.")

            # fallback to nn.Linear
            self.weight.requires_grad_(True)
            if self.bias is not None:
                self.bias.requires_grad_(True)
            # lora_a and lora_b are None, so no need to set requires_grad


class LoRA_BA_Term(nn.Module):
    """
    A linear layer that implements the LoRA (Low-Rank Adaptation) technique. Only includes the addition term, i.e. BA, without W.
    """

    def __init__(
        self, in_features: int, out_features: int, r: int, alpha: float = None
    ):
        super().__init__()
        self.lora_a = nn.Parameter(torch.empty((r, in_features)))
        self.lora_b = nn.Parameter(torch.empty((out_features, r)))

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
            nn.init.zeros_(self.lora_b)

        self.scaling = 1.0 if alpha is None else alpha / r

    def forward(self, x: torch.Tensor):
        return (x @ self.lora_a.T @ self.lora_b.T) * self.scaling
