"""LightningDiT with a TRA head. Parameter names match the released EMA weights."""

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sincos_embedding(width, side):
    x, y = np.meshgrid(np.arange(side, dtype=np.float32), np.arange(side, dtype=np.float32))
    frequency = 1.0 / 10000 ** (np.arange(width // 4, dtype=np.float64) / (width / 4))
    angles = [coordinate.reshape(-1, 1) * frequency for coordinate in (x, y)]
    values = np.concatenate([f(axis) for axis in angles for f in (np.sin, np.cos)], axis=1)
    return torch.from_numpy(values).float().unsqueeze(0)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, hidden_size, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class RotaryEmbedding(nn.Module):
    """Two-dimensional RoPE, with adjacent channel pairs rotated together."""

    def __init__(self, head_dim, side):
        super().__init__()
        axis_dim = head_dim // 2
        inverse = 1.0 / (10000 ** (torch.arange(0, axis_dim, 2).float() / axis_dim))
        position = torch.arange(side) / side * side
        angle = torch.outer(position, inverse).repeat_interleave(2, dim=-1)
        row = angle[:, None, :].expand(side, side, axis_dim)
        column = angle[None, :, :].expand(side, side, axis_dim)
        angle = torch.cat((row, column), dim=-1).reshape(side * side, head_dim)
        self.register_buffer("freqs_cos", angle.cos())
        self.register_buffer("freqs_sin", angle.sin())

    def forward(self, x):
        even, odd = x[..., ::2], x[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
        return x * self.freqs_cos + rotated * self.freqs_sin


class RMSNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x):
        value = x.float()
        inverse_rms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)
        return (value * inverse_rms).to(x.dtype) * self.weight


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, t):
        frequencies = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128).to(t.device)
        angle = (t * 1000).float()[:, None] * frequencies[None]
        return self.mlp(torch.cat((angle.cos(), angle.sin()), dim=-1))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, y, force_drop_ids=None):
        if force_drop_ids is not None:
            y = torch.where(force_drop_ids == 1, self.num_classes, y)
        elif self.training and self.dropout_prob > 0:
            drop = torch.rand(y.shape[0], device=y.device) < self.dropout_prob
            y = torch.where(drop, self.num_classes, y)
        return self.embedding_table(y)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.num_heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.q_norm = RMSNorm(width // heads)
        self.k_norm = RMSNorm(width // heads)
        self.proj = nn.Linear(width, width)

    def forward(self, x, rope):
        batch, tokens, width = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.num_heads, width // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(rope(self.q_norm(q)), rope(self.k_norm(k)), v)
        return self.proj(attended.transpose(1, 2).reshape(batch, tokens, width))


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.w12 = nn.Linear(width, 2 * hidden)
        self.w3 = nn.Linear(hidden, width)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    def __init__(self, width, heads, mlp_ratio):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.attn = Attention(width, heads)
        self.mlp = SwiGLU(width, int((2 / 3) * int(width * mlp_ratio)))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, x, cond, rope):
        s1, a1, g1, s2, a2, g2 = self.adaLN_modulation(cond).chunk(6, dim=1)
        x = x + g1[:, None] * self.attn(modulate(self.norm1(x), s1, a1), rope)
        return x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, a2))


class FinalLayer(nn.Module):
    def __init__(self, width, output_width):
        super().__init__()
        self.norm_final = RMSNorm(width)
        self.linear = nn.Linear(width, output_width)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))

    def forward(self, x, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class LatentDiT(nn.Module):
    """The default configuration is the f16d32 ImageNet-256 LightningDiT-XL/1."""

    def __init__(
        self, depth=28, hidden_size=1152, num_heads=16, latent_shape=(32, 16, 16),
        latent_patch_size=1, num_classes=1000, class_dropout_prob=0.1, mlp_ratio=4.0,
        z_channels_low=16, align_layer=7, use_checkpoint=False,
    ):
        super().__init__()
        channels, height, width = latent_shape
        if height != width or height % latent_patch_size:
            raise ValueError("The latent grid must be square and divisible by the patch size.")
        if hidden_size % (4 * num_heads) or not 0 <= align_layer < depth:
            raise ValueError("Use a head dimension divisible by four and a valid TRA block index.")
        self.latent_channels = channels
        self.z_channels_low = z_channels_low
        self.align_layer = align_layer
        self.use_checkpoint = use_checkpoint
        self.latent_patch_size = latent_patch_size
        self.grid_size = height // latent_patch_size
        self.num_classes = num_classes
        self.x_embedder = PatchEmbed(channels, hidden_size, latent_patch_size)
        # The EMA stores this fixed table. The RoPE model uses feat_rope in attention.
        self.pos_embed = nn.Parameter(sincos_embedding(hidden_size, self.grid_size), requires_grad=False)
        self.feat_rope = RotaryEmbedding(hidden_size // num_heads, self.grid_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.blocks = nn.ModuleList([Block(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, channels * latent_patch_size**2)
        self.repa_proj = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=1e-6), nn.Linear(hidden_size, hidden_size),
            nn.SiLU(), nn.Linear(hidden_size, z_channels_low * latent_patch_size**2),
        )
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        patch_weight = self.x_embedder.proj.weight
        nn.init.xavier_uniform_(patch_weight.view(patch_weight.shape[0], -1))
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        final_projections = [self.final_layer.adaLN_modulation[-1], self.final_layer.linear]
        for layer in [block.adaLN_modulation[-1] for block in self.blocks] + final_projections:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def unpatchify(self, tokens, channels):
        p, side = self.latent_patch_size, self.grid_size
        patches = tokens.reshape(tokens.shape[0], side, side, p, p, channels)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(tokens.shape[0], channels, side * p, side * p)

    def conditioning(self, t, y, force_drop_ids):
        return self.t_embedder(t) + self.y_embedder(y, force_drop_ids)

    def forward(self, x, t, y, force_drop_ids=None, return_repa=False):
        """Predict flow velocity and, with return_repa=True, the TRA skeleton direction."""
        cond = self.conditioning(t, y, force_drop_ids)
        tokens = self.x_embedder(x)
        for index, block in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                tokens = checkpoint(block, tokens, cond, self.feat_rope, use_reentrant=False)
            else:
                tokens = block(tokens, cond, self.feat_rope)
            if return_repa and index == self.align_layer:
                intermediate = tokens
        velocity = self.unpatchify(self.final_layer(tokens, cond), self.latent_channels)
        if return_repa:
            direction = self.unpatchify(self.repa_proj(intermediate), self.z_channels_low)
            return velocity, direction
        return velocity
