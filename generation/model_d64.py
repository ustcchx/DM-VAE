"""LightningDiT with the timestep normalization used by the f16d64 checkpoint."""

import torch

from .model import LatentDiT


class LatentDiT64(LatentDiT):
    """ImageNet-256 DiT-XL/1 with a 64-channel latent and a 32-channel TRA head."""

    def __init__(self, *, latent_shape=(64, 16, 16), z_channels_low=32, **kwargs):
        super().__init__(latent_shape=latent_shape, z_channels_low=z_channels_low, **kwargs)

    def conditioning(self, t, y, force_drop_ids):
        time = self.t_embedder(t)
        inverse_rms = torch.rsqrt(time.float().square().mean(dim=-1, keepdim=True) + 1e-6)
        return time * inverse_rms.to(time.dtype) + self.y_embedder(y, force_drop_ids)
