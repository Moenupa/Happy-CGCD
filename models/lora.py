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
        bias: bool,
        lora_rank: int = 0,
        lora_alpha: int = None,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight.requires_grad = False

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.bias.requires_grad = False
        else:
            self.bias = None

        self.lora_rank = lora_rank
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
        if r > 0:
            self.scaling = 1.0 if alpha is None else alpha / r
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

    def forward(self, x: torch.Tensor):
        result = nn.functional.linear(x, self.weight, bias=self.bias)
        if self.lora_rank > 0:
            result += (
                self.lora_dropout(x) @ self.lora_a.T @ self.lora_b.T
            ) * self.scaling
        return result

    # def load_state_dict(self, state_dict: dict, strict: bool = True):
    #     # Filter out unexpected LoRA keys (if loading from nn.Linear)
    #     own_state = self.state_dict()
    #     missing_keys = []
    #     unexpected_keys = []

    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             try:
    #                 own_state[name].copy_(param)
    #             except Exception as e:
    #                 raise RuntimeError(
    #                     f"While copying the parameter named {name}, "
    #                     f"whose dimensions in the model are {own_state[name].size()} and "
    #                     f"whose dimensions in the checkpoint are {param.size()}, "
    #                     f"an exception occurred: {e}"
    #                 )
    #         else:
    #             if strict:
    #                 unexpected_keys.append(name)

    #     if strict:
    #         for name in own_state:
    #             if name not in state_dict:
    #                 missing_keys.append(name)

    #         if unexpected_keys or missing_keys:
    #             raise RuntimeError(
    #                 f"Error(s) in loading state_dict for {self.__class__.__name__}:\n"
    #                 f"Missing keys: {missing_keys}\n"
    #                 f"Unexpected keys: {unexpected_keys}"
    #             )


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


if __name__ == "__main__":
    # drop-in replacement test
    lora = Linear(10, 5, bias=True, lora_rank=2, lora_alpha=1.0, lora_dropout=0.1)
    linear = nn.Linear(10, 5, bias=True)
    linear.weight.data = lora.weight.data
    linear.bias.data = lora.bias.data

    x = torch.randn(3, 10)
    lora_output = lora(x)
    linear_output = linear(x)
    assert torch.allclose(
        lora_output, linear_output, atol=1e-6
    ), "Outputs do not match!"

    # load_state_dict test, show linear.pth can be loaded as lora.pth
    lora.load_state_dict(linear.state_dict(), strict=False)
    lora_output = lora(x)
    assert torch.allclose(
        lora_output, linear_output, atol=1e-6
    ), "Outputs do not match after loading state_dict!"
