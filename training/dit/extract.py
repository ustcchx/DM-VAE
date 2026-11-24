"""Cache frozen DM-VAE latents and TRA supervision from an ImageFolder train split."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset

from .cache import file_digest


def center_crop(image):
    # ADM's ImageNet preprocessing, adapted from OpenAI guided-diffusion.
    while min(image.size) >= 512:
        image = image.resize(tuple(side // 2 for side in image.size), Image.Resampling.BOX)
    scale = 256 / min(image.size)
    image = image.resize(tuple(round(scale * side) for side in image.size), Image.Resampling.BICUBIC)
    x, y = (image.width - 256) // 2, (image.height - 256) // 2
    array = np.array(image.crop((x, y, x + 256, y + 256)), dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1)


def atomic_json(data, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def image_index_digest(data):
    digest = hashlib.sha256()
    for filename, label in data.samples:
        path = Path(filename)
        metadata = path.stat()
        record = (str(path.relative_to(data.root)), label, metadata.st_size, metadata.st_mtime_ns)
        digest.update(json.dumps(record).encode() + b"\n")
    return digest.hexdigest()


def measure_stats(output, shards, count, high):
    selected = np.sort(np.random.RandomState(0).choice(count, min(10000, count), replace=False))
    total, square = torch.zeros(high, dtype=torch.float64), torch.zeros(high, dtype=torch.float64)
    pixels = 0
    for shard in shards:
        start = shard["start_index"]
        lo, hi = np.searchsorted(selected, [start, start + shard["count"]])
        rows = torch.from_numpy(selected[lo:hi] - start)
        if len(rows):
            with safe_open(str(output / shard["file"]), framework="pt") as reader:
                values = reader.get_tensor("latents")[rows].double()
            total += values.sum(dim=(0, 2, 3))
            square += values.square().sum(dim=(0, 2, 3))
            pixels += values.numel() // high
    mean = total / pixels
    std = ((square - total * mean) / (pixels - 1)).clamp_min(0).sqrt()
    return {"channel_means": mean.float().tolist(), "channel_std": std.float().tolist()}


@torch.inference_mode()
def extract(args):
    from torchvision.datasets import ImageFolder
    from dmvae import load_vae

    if args.batch_size < 1 or args.shard_size < 1 or args.shard_size % args.batch_size:
        raise ValueError("--shard-size must be a positive multiple of --batch-size.")
    if args.workers < 0:
        raise ValueError("--workers must be nonnegative.")
    data = ImageFolder(args.images, transform=center_crop)
    if len(data.classes) > 1000:
        raise ValueError("The DiT supports up to 1000 class indices.")
    vae_sha = file_digest(args.vae)
    vae = load_vae(args.vae).requires_grad_(False)
    high, low = vae.z_channels_high, vae.z_channels_low
    if (high, low) not in ((32, 16), (64, 32)):
        raise ValueError("Use a foundation-model-free f16d32/low16 or f16d64/low32 tokenizer.")
    stats = None
    if args.stats:
        stats = json.loads(args.stats.read_text())
        expected = stats["vae_sha256"]
        if vae_sha not in ([expected] if isinstance(expected, str) else expected):
            raise ValueError("Provided channel statistics belong to a different VAE checkpoint.")
    contract = dict(
        vae_sha256=vae_sha, dataset=image_index_digest(data), shard_size=args.shard_size,
        stats_sha256=file_digest(args.stats) if args.stats else None,
        seed=args.seed, batch_size=args.batch_size, device=args.device,
    )
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    config_path = args.output / "extraction.json"
    if config_path.exists() and json.loads(config_path.read_text()) != contract:
        raise ValueError("Existing features were created with different inputs. Choose another directory.")
    if not config_path.exists():
        if list(args.output.glob("shard-*.safetensors")):
            raise ValueError("Found feature shards without their extraction settings.")
        atomic_json(contract, config_path)
    manifest_path = args.output / "manifest.json"
    previous_shards = {}
    if manifest_path.exists():
        previous_shards = {item["file"]: item["sha256"] for item in json.loads(manifest_path.read_text())["shards"]}
    shards = []
    count = 0
    while count < len(data):
        path = args.output / f"shard-{len(shards):05d}.safetensors"
        if not path.exists():
            break
        size = min(args.shard_size, len(data) - count)
        with safe_open(str(path), framework="pt") as reader:
            if reader.metadata().get("fingerprint") != fingerprint or reader.get_slice("latents").get_shape() != [size, high, 16, 16]:
                raise ValueError(f"Incomplete or incompatible feature shard: {path}")
        digest = file_digest(path)
        if path.name in previous_shards and digest != previous_shards[path.name]:
            raise ValueError(f"Stored feature data changed: {path}")
        shards.append(dict(file=path.name, count=size, start_index=count, bytes=path.stat().st_size, sha256=digest))
        count += size

    if count < len(data):
        device = torch.device(args.device)
        vae.to(device)
        loader = DataLoader(Subset(data, range(count, len(data))), batch_size=args.batch_size,
                            num_workers=args.workers, shuffle=False)
        chunks = {}
        start = count
        for images, labels in loader:
            tensors = {"labels": labels, "indices": torch.arange(count, count + len(labels))}
            images = images.to(device)
            rng = torch.Generator().manual_seed(args.seed + count)
            for suffix, views in (("", images), ("_flip", images.flip(-1))):
                posterior = vae.encode(views)
                mode = posterior.mode()
                noise = torch.randn(mode.shape, dtype=torch.float32, generator=rng).to(device)
                latent = mode + (0.5 * posterior.logvar).exp() * noise
                target = vae.bottleneck_encode(mode)
                if not (latent.isfinite().all() and target.isfinite().all()):
                    raise FloatingPointError(f"Non-finite VAE output in the batch starting at image {count}.")
                tensors["latents" + suffix] = latent.float().cpu()
                tensors["low_targets" + suffix] = target.float().cpu()
            for key, value in tensors.items():
                chunks.setdefault(key, []).append(value)
            count += len(labels)
            if count - start == args.shard_size or count == len(data):
                path = args.output / f"shard-{len(shards):05d}.safetensors"
                temporary = path.with_suffix(".tmp")
                save_file({k: torch.cat(v) for k, v in chunks.items()}, str(temporary),
                          metadata={"fingerprint": fingerprint})
                temporary.replace(path)
                shards.append(dict(file=path.name, count=count - start, start_index=start,
                                   bytes=path.stat().st_size, sha256=file_digest(path)))
                print(f"Cached {count}/{len(data)} images", flush=True)
                chunks, start = {}, count

    if previous_shards and {row["file"] for row in shards} != set(previous_shards):
        raise ValueError("Feature inventory changed.")
    if stats is None:
        stats = measure_stats(args.output, shards, len(data), high)
    stats = {key: stats[key] for key in ("channel_means", "channel_std")}
    for key, values in stats.items():
        array = np.asarray(values)
        if array.shape != (high,) or not np.isfinite(array).all() or (key == "channel_std" and (array <= 0).any()):
            raise ValueError(f"Invalid {key}.")
    atomic_json({"vae_sha256": vae_sha, "skeleton_channels": low, **stats}, args.output / "stats.json")
    atomic_json(dict(schema="dflow_paired_latent_cache/v1", state="complete", posterior="sample",
                     repa_target="bottleneck_encode(posterior.mode())", dtype="float32",
                     latent_shape=[high, 16, 16], target_shape=[low, 16, 16], vae_ckpt_sha256=vae_sha,
                     count=len(data), classes=data.classes, stats=stats, shards=shards), args.output / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="ImageFolder root containing one directory per class")
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, help="Use matching published statistics (default: estimate from this cache)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".extract.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        extract(args)


if __name__ == "__main__":
    main()
