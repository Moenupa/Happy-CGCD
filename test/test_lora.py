import torch
import torch.nn as nn
from models import lora
from models import lora_vision_transformer as vits


def test_lora_linear():
    loralinear = lora.Linear(
        10, 5, bias=True, lora_rank=2, lora_alpha=1.0, lora_dropout=0.1
    )
    linear = nn.Linear(10, 5, bias=True)
    linear.weight.data = loralinear.weight.data
    linear.bias.data = loralinear.bias.data

    x = torch.randn(3, 10)
    lora_output = loralinear(x)
    linear_output = linear(x)
    assert torch.allclose(
        lora_output, linear_output, atol=1e-6
    ), "Outputs do not match!"

    # load_state_dict test, show linear.pth can be loaded as lora.pth
    loralinear.load_state_dict(linear.state_dict(), strict=False)
    lora_output = loralinear(x)
    assert torch.allclose(
        lora_output, linear_output, atol=1e-6
    ), "Outputs do not match after loading state_dict!"


def test_lora_grad():
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
        if hasattr(m, "fix_grad"):
            if m.weight.requires_grad:
                m.fix_grad()

    EXPECTED_GRAD_DICT = {
        "cls_token": False,
        "pos_embed": False,
        "patch_embed.proj.weight": False,
        "patch_embed.proj.bias": False,
        "blocks.0.norm1.weight": False,
        "blocks.0.norm1.bias": False,
        "blocks.0.attn.qkv.weight": False,
        "blocks.0.attn.qkv.bias": False,
        "blocks.0.attn.qkv.lora_a": False,
        "blocks.0.attn.qkv.lora_b": False,
        "blocks.0.attn.proj.weight": False,
        "blocks.0.attn.proj.bias": False,
        "blocks.0.attn.proj.lora_a": False,
        "blocks.0.attn.proj.lora_b": False,
        "blocks.0.norm2.weight": False,
        "blocks.0.norm2.bias": False,
        "blocks.0.mlp.fc1.weight": False,
        "blocks.0.mlp.fc1.bias": False,
        "blocks.0.mlp.fc1.lora_a": False,
        "blocks.0.mlp.fc1.lora_b": False,
        "blocks.0.mlp.fc2.weight": False,
        "blocks.0.mlp.fc2.bias": False,
        "blocks.0.mlp.fc2.lora_a": False,
        "blocks.0.mlp.fc2.lora_b": False,
        "blocks.1.norm1.weight": False,
        "blocks.1.norm1.bias": False,
        "blocks.1.attn.qkv.weight": False,
        "blocks.1.attn.qkv.bias": False,
        "blocks.1.attn.qkv.lora_a": False,
        "blocks.1.attn.qkv.lora_b": False,
        "blocks.1.attn.proj.weight": False,
        "blocks.1.attn.proj.bias": False,
        "blocks.1.attn.proj.lora_a": False,
        "blocks.1.attn.proj.lora_b": False,
        "blocks.1.norm2.weight": False,
        "blocks.1.norm2.bias": False,
        "blocks.1.mlp.fc1.weight": False,
        "blocks.1.mlp.fc1.bias": False,
        "blocks.1.mlp.fc1.lora_a": False,
        "blocks.1.mlp.fc1.lora_b": False,
        "blocks.1.mlp.fc2.weight": False,
        "blocks.1.mlp.fc2.bias": False,
        "blocks.1.mlp.fc2.lora_a": False,
        "blocks.1.mlp.fc2.lora_b": False,
        "blocks.2.norm1.weight": False,
        "blocks.2.norm1.bias": False,
        "blocks.2.attn.qkv.weight": False,
        "blocks.2.attn.qkv.bias": False,
        "blocks.2.attn.qkv.lora_a": False,
        "blocks.2.attn.qkv.lora_b": False,
        "blocks.2.attn.proj.weight": False,
        "blocks.2.attn.proj.bias": False,
        "blocks.2.attn.proj.lora_a": False,
        "blocks.2.attn.proj.lora_b": False,
        "blocks.2.norm2.weight": False,
        "blocks.2.norm2.bias": False,
        "blocks.2.mlp.fc1.weight": False,
        "blocks.2.mlp.fc1.bias": False,
        "blocks.2.mlp.fc1.lora_a": False,
        "blocks.2.mlp.fc1.lora_b": False,
        "blocks.2.mlp.fc2.weight": False,
        "blocks.2.mlp.fc2.bias": False,
        "blocks.2.mlp.fc2.lora_a": False,
        "blocks.2.mlp.fc2.lora_b": False,
        "blocks.3.norm1.weight": False,
        "blocks.3.norm1.bias": False,
        "blocks.3.attn.qkv.weight": False,
        "blocks.3.attn.qkv.bias": False,
        "blocks.3.attn.qkv.lora_a": False,
        "blocks.3.attn.qkv.lora_b": False,
        "blocks.3.attn.proj.weight": False,
        "blocks.3.attn.proj.bias": False,
        "blocks.3.attn.proj.lora_a": False,
        "blocks.3.attn.proj.lora_b": False,
        "blocks.3.norm2.weight": False,
        "blocks.3.norm2.bias": False,
        "blocks.3.mlp.fc1.weight": False,
        "blocks.3.mlp.fc1.bias": False,
        "blocks.3.mlp.fc1.lora_a": False,
        "blocks.3.mlp.fc1.lora_b": False,
        "blocks.3.mlp.fc2.weight": False,
        "blocks.3.mlp.fc2.bias": False,
        "blocks.3.mlp.fc2.lora_a": False,
        "blocks.3.mlp.fc2.lora_b": False,
        "blocks.4.norm1.weight": False,
        "blocks.4.norm1.bias": False,
        "blocks.4.attn.qkv.weight": False,
        "blocks.4.attn.qkv.bias": False,
        "blocks.4.attn.qkv.lora_a": False,
        "blocks.4.attn.qkv.lora_b": False,
        "blocks.4.attn.proj.weight": False,
        "blocks.4.attn.proj.bias": False,
        "blocks.4.attn.proj.lora_a": False,
        "blocks.4.attn.proj.lora_b": False,
        "blocks.4.norm2.weight": False,
        "blocks.4.norm2.bias": False,
        "blocks.4.mlp.fc1.weight": False,
        "blocks.4.mlp.fc1.bias": False,
        "blocks.4.mlp.fc1.lora_a": False,
        "blocks.4.mlp.fc1.lora_b": False,
        "blocks.4.mlp.fc2.weight": False,
        "blocks.4.mlp.fc2.bias": False,
        "blocks.4.mlp.fc2.lora_a": False,
        "blocks.4.mlp.fc2.lora_b": False,
        "blocks.5.norm1.weight": False,
        "blocks.5.norm1.bias": False,
        "blocks.5.attn.qkv.weight": False,
        "blocks.5.attn.qkv.bias": False,
        "blocks.5.attn.qkv.lora_a": False,
        "blocks.5.attn.qkv.lora_b": False,
        "blocks.5.attn.proj.weight": False,
        "blocks.5.attn.proj.bias": False,
        "blocks.5.attn.proj.lora_a": False,
        "blocks.5.attn.proj.lora_b": False,
        "blocks.5.norm2.weight": False,
        "blocks.5.norm2.bias": False,
        "blocks.5.mlp.fc1.weight": False,
        "blocks.5.mlp.fc1.bias": False,
        "blocks.5.mlp.fc1.lora_a": False,
        "blocks.5.mlp.fc1.lora_b": False,
        "blocks.5.mlp.fc2.weight": False,
        "blocks.5.mlp.fc2.bias": False,
        "blocks.5.mlp.fc2.lora_a": False,
        "blocks.5.mlp.fc2.lora_b": False,
        "blocks.6.norm1.weight": False,
        "blocks.6.norm1.bias": False,
        "blocks.6.attn.qkv.weight": False,
        "blocks.6.attn.qkv.bias": False,
        "blocks.6.attn.qkv.lora_a": False,
        "blocks.6.attn.qkv.lora_b": False,
        "blocks.6.attn.proj.weight": False,
        "blocks.6.attn.proj.bias": False,
        "blocks.6.attn.proj.lora_a": False,
        "blocks.6.attn.proj.lora_b": False,
        "blocks.6.norm2.weight": False,
        "blocks.6.norm2.bias": False,
        "blocks.6.mlp.fc1.weight": False,
        "blocks.6.mlp.fc1.bias": False,
        "blocks.6.mlp.fc1.lora_a": False,
        "blocks.6.mlp.fc1.lora_b": False,
        "blocks.6.mlp.fc2.weight": False,
        "blocks.6.mlp.fc2.bias": False,
        "blocks.6.mlp.fc2.lora_a": False,
        "blocks.6.mlp.fc2.lora_b": False,
        "blocks.7.norm1.weight": False,
        "blocks.7.norm1.bias": False,
        "blocks.7.attn.qkv.weight": False,
        "blocks.7.attn.qkv.bias": False,
        "blocks.7.attn.qkv.lora_a": False,
        "blocks.7.attn.qkv.lora_b": False,
        "blocks.7.attn.proj.weight": False,
        "blocks.7.attn.proj.bias": False,
        "blocks.7.attn.proj.lora_a": False,
        "blocks.7.attn.proj.lora_b": False,
        "blocks.7.norm2.weight": False,
        "blocks.7.norm2.bias": False,
        "blocks.7.mlp.fc1.weight": False,
        "blocks.7.mlp.fc1.bias": False,
        "blocks.7.mlp.fc1.lora_a": False,
        "blocks.7.mlp.fc1.lora_b": False,
        "blocks.7.mlp.fc2.weight": False,
        "blocks.7.mlp.fc2.bias": False,
        "blocks.7.mlp.fc2.lora_a": False,
        "blocks.7.mlp.fc2.lora_b": False,
        "blocks.8.norm1.weight": False,
        "blocks.8.norm1.bias": False,
        "blocks.8.attn.qkv.weight": False,
        "blocks.8.attn.qkv.bias": False,
        "blocks.8.attn.qkv.lora_a": False,
        "blocks.8.attn.qkv.lora_b": False,
        "blocks.8.attn.proj.weight": False,
        "blocks.8.attn.proj.bias": False,
        "blocks.8.attn.proj.lora_a": False,
        "blocks.8.attn.proj.lora_b": False,
        "blocks.8.norm2.weight": False,
        "blocks.8.norm2.bias": False,
        "blocks.8.mlp.fc1.weight": False,
        "blocks.8.mlp.fc1.bias": False,
        "blocks.8.mlp.fc1.lora_a": False,
        "blocks.8.mlp.fc1.lora_b": False,
        "blocks.8.mlp.fc2.weight": False,
        "blocks.8.mlp.fc2.bias": False,
        "blocks.8.mlp.fc2.lora_a": False,
        "blocks.8.mlp.fc2.lora_b": False,
        "blocks.9.norm1.weight": False,
        "blocks.9.norm1.bias": False,
        "blocks.9.attn.qkv.weight": False,
        "blocks.9.attn.qkv.bias": False,
        "blocks.9.attn.qkv.lora_a": False,
        "blocks.9.attn.qkv.lora_b": False,
        "blocks.9.attn.proj.weight": False,
        "blocks.9.attn.proj.bias": False,
        "blocks.9.attn.proj.lora_a": False,
        "blocks.9.attn.proj.lora_b": False,
        "blocks.9.norm2.weight": False,
        "blocks.9.norm2.bias": False,
        "blocks.9.mlp.fc1.weight": False,
        "blocks.9.mlp.fc1.bias": False,
        "blocks.9.mlp.fc1.lora_a": False,
        "blocks.9.mlp.fc1.lora_b": False,
        "blocks.9.mlp.fc2.weight": False,
        "blocks.9.mlp.fc2.bias": False,
        "blocks.9.mlp.fc2.lora_a": False,
        "blocks.9.mlp.fc2.lora_b": False,
        "blocks.10.norm1.weight": True,
        "blocks.10.norm1.bias": True,
        "blocks.10.attn.qkv.weight": False,
        "blocks.10.attn.qkv.bias": False,
        "blocks.10.attn.qkv.lora_a": True,
        "blocks.10.attn.qkv.lora_b": True,
        "blocks.10.attn.proj.weight": False,
        "blocks.10.attn.proj.bias": False,
        "blocks.10.attn.proj.lora_a": True,
        "blocks.10.attn.proj.lora_b": True,
        "blocks.10.norm2.weight": True,
        "blocks.10.norm2.bias": True,
        "blocks.10.mlp.fc1.weight": False,
        "blocks.10.mlp.fc1.bias": False,
        "blocks.10.mlp.fc1.lora_a": True,
        "blocks.10.mlp.fc1.lora_b": True,
        "blocks.10.mlp.fc2.weight": False,
        "blocks.10.mlp.fc2.bias": False,
        "blocks.10.mlp.fc2.lora_a": True,
        "blocks.10.mlp.fc2.lora_b": True,
        "blocks.11.norm1.weight": True,
        "blocks.11.norm1.bias": True,
        "blocks.11.attn.qkv.weight": False,
        "blocks.11.attn.qkv.bias": False,
        "blocks.11.attn.qkv.lora_a": True,
        "blocks.11.attn.qkv.lora_b": True,
        "blocks.11.attn.proj.weight": False,
        "blocks.11.attn.proj.bias": False,
        "blocks.11.attn.proj.lora_a": True,
        "blocks.11.attn.proj.lora_b": True,
        "blocks.11.norm2.weight": True,
        "blocks.11.norm2.bias": True,
        "blocks.11.mlp.fc1.weight": False,
        "blocks.11.mlp.fc1.bias": False,
        "blocks.11.mlp.fc1.lora_a": True,
        "blocks.11.mlp.fc1.lora_b": True,
        "blocks.11.mlp.fc2.weight": False,
        "blocks.11.mlp.fc2.bias": False,
        "blocks.11.mlp.fc2.lora_a": True,
        "blocks.11.mlp.fc2.lora_b": True,
        "norm.weight": False,
        "norm.bias": False,
    }

    for name, m in backbone.named_parameters():
        assert m.requires_grad == EXPECTED_GRAD_DICT[name], (
            f"Parameter {name} has requires_grad={m.requires_grad}, "
            f"expected {EXPECTED_GRAD_DICT[name]}"
        )


if __name__ == "__main__":
    test_lora_grad()
