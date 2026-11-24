"""Rectified-flow velocity and topological representation alignment losses."""
import torch
from torch.nn import functional as F


def flow_inputs(latent, logit_mean=0.0):
    noise = torch.randn_like(latent)
    time = (torch.randn(latent.shape[0], device=latent.device) + logit_mean).sigmoid()
    fraction = time[:, None, None, None]
    return (1 - fraction) * noise + fraction * latent, time, latent - noise


def normalized_skeleton_projector(decode_weight, channel_std):
    """Orthogonally project velocities onto the VAE skeleton in normalized coordinates."""
    weight = decode_weight.detach().cpu().double()
    std = channel_std.detach().cpu().double().reshape(-1)
    if weight.shape != (64, 32, 1, 1) or std.shape != (64,) or not (std > 0).all():
        raise ValueError("Expected a 64 × 32 decoder projection and 64 positive standard deviations.")
    basis, triangular = torch.linalg.qr(weight[:, :, 0, 0] / std[:, None])
    if not torch.isfinite(triangular).all() or torch.linalg.matrix_rank(triangular).item() != 32:
        raise ValueError("The normalized decoder projection must have rank 32.")
    return (basis @ basis.T).float()


def loss_terms(prediction, velocity, skeleton_prediction, skeleton, time=None, projector=None):
    prediction = prediction.float()
    velocity = velocity.float()
    if projector is not None:
        # Down-weighting the residual velocity at early high-noise steps slightly improves gFID, so we use a non-uniform residual_weight for f16d64
        residual_weight = (1 - time.float())[:, None, None, None]
        pred_base = torch.einsum("dc,bchw->bdhw", projector, prediction)
        target_base = torch.einsum("dc,bchw->bdhw", projector, velocity)
        prediction = pred_base + residual_weight * (prediction - pred_base)
        velocity = target_base + residual_weight * (velocity - target_base)

    mse = F.mse_loss(prediction, velocity)
    direction = 1 - F.cosine_similarity(prediction, velocity, dim=1).mean()
    alignment = 1 - F.cosine_similarity(skeleton_prediction.float(), skeleton.float(), dim=1).mean()
    return mse + direction + 0.5 * alignment, mse, direction, alignment
