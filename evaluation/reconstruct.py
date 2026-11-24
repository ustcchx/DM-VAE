"""Export paired image crops and DM-VAE reconstructions for rFID and PSNR."""

import argparse
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from dmvae import load_vae


class ValidationImages(Dataset):
    def __init__(self, root):
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        self.paths = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
            and not p.name.startswith("._") and "__MACOSX" not in p.relative_to(root).parts
        )
        if not self.paths:
            raise ValueError(f"No validation images found in {root}.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            crop = ImageOps.fit(image, (256, 256), method=Image.Resampling.BICUBIC)
            return torch.from_numpy(np.array(crop)).permute(2, 0, 1)


@torch.inference_mode()
def export(args):
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    if args.output.resolve().is_relative_to(args.images.resolve()):
        raise ValueError("Choose an output directory outside the input image tree.")
    dataset = ValidationImages(args.images)
    model = load_vae(args.vae, device=args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        staging = Path(temp) / "pairs"
        (staging / "reference").mkdir(parents=True)
        (staging / "reconstructed").mkdir()
        psnr_sum = 0.0
        count = 0
        for batch in loader:
            images = batch.to(args.device, torch.float32).div(127.5).sub(1)
            latents = model.encode(images).mode()
            outputs = model.decode(latents)
            if not torch.isfinite(outputs).all():
                raise FloatingPointError(f"Non-finite reconstruction at batch starting with image {count}.")
            outputs = outputs.clamp(-1, 1).add(1).mul(127.5).round().byte().cpu()
            errors = (outputs.double() - batch.double()).square().mean(dim=(1, 2, 3))
            psnr_sum += (10 * (255.0 ** 2 / errors).log10()).sum().item()
            for original, reconstructed in zip(batch, outputs):
                name = f"{count:06d}.png"
                for directory, tensor in (("reference", original), ("reconstructed", reconstructed)):
                    pixels = tensor.permute(1, 2, 0).numpy()
                    Image.fromarray(pixels).save(staging / directory / name)
                count += 1
        score = psnr_sum / count
        score_text = f"{score:.5f}" if math.isfinite(score) else "infinity"
        summary = f"{count} paired images\nMean per-image PSNR: {score_text} dB\n"
        (staging / "metrics.txt").write_text(summary)
        os.rename(staging, args.output)
        print(summary, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Validation image tree")
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers must be nonnegative.")
    export(args)


if __name__ == "__main__":
    main()
