"""Generate class-conditional ImageNet-256 images with DM-VAE and LightningDiT."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from .model import LatentDiT
from .sampling import euler_sample, RECIPES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit", required=True, type=Path, help="LightningDiT-XL EMA weights (.safetensors or .pt)")
    parser.add_argument("--vae", required=True, type=Path, help="Corresponding DM-VAE EMA")
    parser.add_argument("--recipe", choices=RECIPES, default="400ep-ig")
    parser.add_argument("--stats", type=Path, help="Channel statistics (defaults to the tokenizer paired with the recipe)")
    parser.add_argument("--output", type=Path, default=Path("samples"))
    parser.add_argument("--count", type=int, default=4, help="Total number of images to generate")
    parser.add_argument("--classes", nargs="+", type=int, help="Cycle through these ImageNet class indices")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0, help="First global sample ID")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1234, help="Sample i uses seed + i")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["bf16", "fp32"], help="Defaults to bf16 on CUDA and fp32 elsewhere")
    args = parser.parse_args()
    if min(args.count, args.steps, args.batch_size) < 1 or args.start_index < 0:
        parser.error("count, steps and batch-size must be positive, and start-index nonnegative")
    if args.classes is not None and any(c < 0 or c >= 1000 for c in args.classes):
        parser.error("ImageNet class indices must be between 0 and 999")
    return args


def verify_vae(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() not in ([expected] if isinstance(expected, str) else expected):
        raise ValueError("The VAE checkpoint does not match the tokenizer that produced these channel statistics.")


def load_stats(path, device, dtype):
    data = json.loads(path.read_text())
    means = torch.tensor(data["channel_means"], device=device, dtype=dtype)
    std = torch.tensor(data["channel_std"], dtype=torch.float64)
    if means.shape not in {(32,), (64,)} or means.shape != std.shape:
        raise ValueError("Expected one mean and one standard deviation per latent channel (32 or 64).")
    if not torch.isfinite(means).all() or not (torch.isfinite(std) & (std > 0)).all():
        raise ValueError("Channel means must be finite and standard deviations must be finite and positive.")
    # The evaluation worker computed reciprocals in double precision before
    # casting to bf16. Prematurely rounding the standard deviations changes the scale.
    scales = std.reciprocal().to(device=device, dtype=dtype)
    return means.view(1, -1, 1, 1), scales.view(1, -1, 1, 1), data["vae_sha256"]


@torch.no_grad()
def main():
    args = parse_args()
    from dmvae import load_state, load_vae

    device = torch.device(args.device)
    precision = args.precision or ("bf16" if device.type == "cuda" else "fp32")
    if precision == "bf16" and device.type not in {"cpu", "cuda"}:
        raise ValueError("Use fp32 on this device.")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16. Select --precision fp32.")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    if args.recipe.startswith("d64-"):
        preset = "f16d64.json"
    elif args.recipe.startswith("dino-"):
        preset = "f16d32-dino.json"
    else:
        preset = "f16d32.json"
    stats_path = args.stats or Path(__file__).with_name("configs") / preset
    means, scales, expected_vae = load_stats(stats_path, device, dtype)
    verify_vae(args.vae, expected_vae)
    vae = load_vae(args.vae, device=device).eval().to(dtype=dtype)
    channels = vae.z_channels_high
    if channels != means.shape[1]:
        raise ValueError("The tokenizer and channel statistics have different latent dimensions.")
    if args.recipe != "unguided":
        expected_shape = {"f16d32.json": (32, 16), "f16d32-dino.json": (32, 12), "f16d64.json": (64, 32)}[preset]
        if (channels, vae.z_channels_low) != expected_shape:
            raise ValueError(f"The {args.recipe} recipe requires a tokenizer with high/low dimensions {expected_shape}.")
    # A meta-device model and memory-mapped weights avoid transient copies of the 2.7 GB XL weights.
    with torch.device("meta"):
        if channels == 64:
            from .model_d64 import LatentDiT64
            model = LatentDiT64()
        else:
            model = LatentDiT(z_channels_low=vae.z_channels_low)
    state = load_state(args.dit, mmap=True)
    model.load_state_dict(state, strict=True, assign=True)
    model = model.float().to(device).eval()
    del state

    ig, cfg = RECIPES[args.recipe]
    args.output.mkdir(parents=True, exist_ok=True)
    end = args.start_index + args.count
    for first in range(args.start_index, end, args.batch_size):
        ids = list(range(first, min(first + args.batch_size, end)))
        classes = [args.classes[i % len(args.classes)] if args.classes else i % 1000 for i in ids]
        labels = torch.tensor(classes, device=device)
        # CPU RNG also permits running example inference on MPS.
        noise_device = device if device.type == "cuda" else torch.device("cpu")
        noise = torch.stack([
            torch.randn((channels, 16, 16), device=noise_device,
                        generator=torch.Generator(device=noise_device).manual_seed(args.seed + i))
            for i in ids
        ]).to(device=device, dtype=dtype)
        context = torch.autocast(device.type, dtype=dtype) if precision == "bf16" else nullcontext()
        with context:
            latent = euler_sample(model, vae, noise, labels, means, scales, ig, cfg, args.steps)
            rgb = vae.decode(latent / scales + means)
        if not torch.isfinite(rgb).all():
            raise FloatingPointError(f"Non-finite RGB values in the batch starting at sample {first}")
        pixels = ((rgb.float() + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for index, label, array in zip(ids, classes, pixels):
            path = args.output / f"{index:06d}_class{label:04d}.png"
            Image.fromarray(array).save(path)
            print(path, flush=True)


if __name__ == "__main__":
    main()
