import torch
import torch.nn as nn

from models import svft
from models import svft_vision_transformer as vits


def test_svft_grad():
    """
    Test if LoRA parameters are trainable.
    """
    backbone = vits.vit_base()

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if "block" in name:
            block_num = int(name.split(".")[1])
            if block_num >= 10:
                m.requires_grad = True

    for name, m in backbone.named_modules():
        if "." not in name:
            continue

        if "block" in name:
            block_num = int(name.split(".")[1])
            if block_num < 10:
                continue

        # a special module-level fix for grad updates
        # only work on LoRA modules, incl. lora.Linear and svft.Linear
        if hasattr(m, "lazy_init"):
            m.lazy_init(
                mask_pattern="banded", off_diag=3, rank=10, fill_orthonormal=True
            )

        if hasattr(m, "fix_grad"):
            m.fix_grad()

    for name, m in backbone.named_parameters():
        print(f'"{name}": {m.requires_grad},')
        # assert m.requires_grad == EXPECTED_GRAD_DICT[name], (
        #     f"Parameter {name} has requires_grad={m.requires_grad}, "
        #     f"expected {EXPECTED_GRAD_DICT[name]}"
        # )


if __name__ == "__main__":
    test_svft_grad()
