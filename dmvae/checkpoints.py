"""Strict loading of original tokenizer weights and release-training checkpoints."""
from collections.abc import Mapping

import torch

from .autoencoder import DMVAE


def load_state(path, mmap=False):
    """Read a state dictionary from a safetensors file or a PyTorch checkpoint."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)


def load_vae(path, device="cpu"):
    """Infer architecture from a local checkpoint and return the tokenizer in eval mode."""
    state = load_state(path)
    if not isinstance(state, Mapping):
        raise ValueError("Expected a state dictionary or a training checkpoint.")
    for wrapper in ("ema_model", "ema", "model", "state_dict"):
        if wrapper in state and isinstance(state[wrapper], Mapping):
            state = state[wrapper]
            break

    weights = {}
    for key, value in state.items():
        if key == "___num_updates___":
            continue
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        if key in weights or not isinstance(value, torch.Tensor):
            raise ValueError(f"Invalid tokenizer parameter {key!r}.")
        weights[key] = value

    legacy = "bottleneck_encode.0.weight" in weights
    projection = "bottleneck_encode.0.weight" if legacy else "bottleneck_encode.weight"
    if projection not in weights:
        raise ValueError("Missing DM-VAE bottleneck. Provide a dual-manifold tokenizer checkpoint.")
    low, high, kh, kw = weights[projection].shape
    if (kh, kw) != (1, 1):
        raise ValueError("Expected a 1×1 tokenizer bottleneck.")
    vf_dim = weights["vf_proj.weight"].shape[0] if "vf_proj.weight" in weights else None
    model = DMVAE(high, low, legacy_bottleneck=legacy, vf_dim=vf_dim)
    model.load_state_dict(weights, strict=True)
    return model.eval().to(device)
