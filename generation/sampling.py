"""Uniform Euler integration with DM-VAE's IG_TS and optional interval CFG."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Guidance:
    scale: float = 1.0
    start: float = 0.0
    end: float = 1.0

    def applies(self, t):
        return self.scale != 1.0 and self.start <= t < self.end


# Checkpoint-specific ImageNet-256 sampling parameters.
RECIPES = {
    "64ep-ig": (Guidance(2.6, 0.0, 0.3), Guidance()),
    "64ep-ig-cfg": (Guidance(1.5, 0.0, 0.3), Guidance(2.65, 0.3, 0.7)),
    "400ep-ig": (Guidance(1.9, 0.0, 0.55), Guidance()),
    "400ep-ig-cfg": (Guidance(1.5, 0.0, 0.55), Guidance(1.3, 0.15, 1.0)),
    "dino-400ep-ig": (Guidance(1.8, 0.05, 0.80), Guidance()),
    "dino-400ep-ig-cfg": (Guidance(1.4, 0.05, 0.95), Guidance(1.35, 0.15, 1.0)),
    "d64-400ep-ig": (Guidance(1.9, 0.0, 0.60), Guidance()),
    "d64-400ep-ig-cfg": (Guidance(1.5, 0.0, 0.60), Guidance(1.35, 0.15, 1.0)),
    "unguided": (Guidance(), Guidance()),
}


def skeleton_guidance(vae, z_t, velocity, direction, time_left, scale, means, scales):
    clean = (z_t + time_left * velocity) / scales + means
    skeleton = vae.bottleneck_encode(clean)
    base = vae.bottleneck_decode(skeleton)
    detail = clean - base

    # The TRA head is trained with cosine similarity. Supply the norm from
    # the final prediction at the same spatial location before lifting it.
    length = torch.norm(skeleton.float(), p=2, dim=1, keepdim=True) + 1e-6
    unit = F.normalize(direction.float(), p=2, dim=1, eps=1e-6)
    intermediate = vae.bottleneck_decode((unit * length).to(z_t.dtype))
    guided = base + (scale - 1.0) * (base - intermediate) + detail
    return ((guided - means) * scales - z_t) / time_left


@torch.no_grad()
def euler_sample(model, vae, noise, labels, means, scales, ig, cfg, steps=250):
    """Inputs and return use normalized DiT coordinates, with t=0 at noise."""
    if steps < 1:
        raise ValueError("steps must be positive")
    state = noise.float()
    null = torch.full_like(labels, model.num_classes)
    dt = 1.0 / steps
    for step in range(steps):
        t = step / steps
        remaining = 1.0 - t
        z_t = state.to(noise.dtype)
        times = torch.full((len(labels),), t, device=noise.device, dtype=torch.float32).to(noise.dtype)
        if ig.applies(t) and remaining > 1e-4:
            velocity, direction = model(z_t, times, labels, return_repa=True)
            velocity = skeleton_guidance(vae, z_t, velocity, direction, remaining, ig.scale, means, scales)
        else:
            velocity = model(z_t, times, labels)
        if cfg.applies(t):
            unconditional = model(z_t, times, null).float()
            velocity = unconditional + cfg.scale * (velocity.float() - unconditional)
        state = state + velocity.float() * dt
    return state.to(noise.dtype)
