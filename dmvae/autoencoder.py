"""DM-VAE f16 encoder, high decoder and low decoder.

The convolutional backbone follows the CompVis latent-diffusion autoencoder.
Parameter names match the original training and EMA checkpoints.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape

        q = q.view(b, 1, c, h * w).transpose(-2, -1).contiguous()  # [B, 1, H*W, C]
        k = k.view(b, 1, c, h * w).transpose(-2, -1).contiguous()  # [B, 1, H*W, C]
        v = v.view(b, 1, c, h * w).transpose(-2, -1).contiguous()  # [B, 1, H*W, C]

        h_ = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        h_ = h_.transpose(-2, -1).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels_high=32,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.norm_out = Normalize(block_in)

        out_dim_high = 2 * z_channels_high if double_z else z_channels_high
        self.conv_out_high = torch.nn.Conv2d(block_in, out_dim_high, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h_high = self.conv_out_high(h)

        return h_high


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        give_pre_end=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        if len(attn_resolutions) > 0:
            self.mid.attn_1 = AttnBlock(block_in)
        else:
            self.mid.attn_1 = nn.Identity()
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h



class DiagonalGaussianDistribution:
    def __init__(self, moments, max_logvar):
        self.mean, logvar = moments.chunk(2, dim=1)
        self.logvar = logvar.clamp(-30, max_logvar)

    def mode(self):
        return self.mean

    def sample(self, generator=None):
        noise = torch.randn(
            self.mean.shape, device=self.mean.device, dtype=self.mean.dtype, generator=generator
        )
        return self.mean + (0.5 * self.logvar).exp() * noise


class DMVAE(nn.Module):
    """Dual-manifold VAE for images in [-1, 1], with channel-first latents."""

    def __init__(self, z_channels_high=32, z_channels_low=16, *, vf_dim=None,
                 legacy_bottleneck=False):
        super().__init__()
        if not 0 < z_channels_low <= z_channels_high:
            raise ValueError("Expected 0 < low channels <= high channels.")
        self.z_channels_high = z_channels_high
        self.z_channels_low = z_channels_low
        self.max_logvar = 20.0 if legacy_bottleneck else 10.0
        self.encoder = Encoder(z_channels_high=z_channels_high)
        self.decoder = Decoder(ch=128, num_res_blocks=2, z_channels=z_channels_high,
                               attn_resolutions=(16,))
        self.decoder_low = Decoder(ch=64, num_res_blocks=1, z_channels=z_channels_low,
                                   attn_resolutions=())
        self.quant_conv = nn.Conv2d(2 * z_channels_high, 2 * z_channels_high, 1)
        self.post_quant_conv = nn.Conv2d(z_channels_high, z_channels_high, 1)
        self.post_quant_conv_low = nn.Conv2d(z_channels_low, z_channels_low, 1)
        if legacy_bottleneck:
            self.bottleneck_encode = nn.Sequential(nn.Conv2d(z_channels_high, z_channels_low, 1),
                                                    nn.SiLU())
            self.bottleneck_decode = nn.Conv2d(z_channels_low, z_channels_high, 1)
        else:
            self.bottleneck_encode = nn.Conv2d(z_channels_high, z_channels_low, 1, bias=False)
            self.bottleneck_decode = nn.Conv2d(z_channels_low, z_channels_high, 1, bias=False)
            with torch.no_grad():
                basis = self.bottleneck_decode.weight[:, :, 0, 0]
                nn.init.orthogonal_(basis)
                self.bottleneck_encode.weight[:, :, 0, 0].copy_(basis.T)
        if vf_dim is not None:
            self.vf_proj = nn.Conv2d(z_channels_high, vf_dim, 1)

    def encode(self, x):
        """Return the posterior before any generator-specific normalization."""
        if x.ndim != 4 or x.shape[1] != 3 or any(s < 16 or s % 16 for s in x.shape[-2:]):
            raise ValueError("Expected RGB NCHW input with height/width positive multiples of 16.")
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)), self.max_logvar)

    def decode(self, z):
        """Reconstruct an image from a high-dimensional raw latent."""
        return self.decoder(self.post_quant_conv(z))

    def decode_low(self, z_low):
        """Reconstruct an image from the compressed skeleton."""
        return self.decoder_low(self.post_quant_conv_low(z_low))

    def forward(self, images):
        """Return both reconstructions and posterior terms for tokenizer training."""
        posterior = self.encode(images)
        z = posterior.sample()
        low = self.bottleneck_encode(z)
        mu = posterior.mode()
        mu_hat = self.bottleneck_decode(self.bottleneck_encode(mu))
        return self.decode(z), self.decode_low(low), mu, posterior.logvar, mu_hat
