"""Reconstruct one image through the full and skeleton decoders."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .checkpoints import load_vae


def save_image(tensor, path):
    rgb = tensor.detach().float().clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
    Image.fromarray(rgb[0].permute(1, 2, 0).cpu().numpy()).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for input.png, high.png, low.png")
    parser.add_argument("--size", type=int, default=256, help="Square center crop after resizing the shorter side")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.size < 16 or args.size % 16:
        parser.error("--size must be a positive multiple of 16")
    with Image.open(args.input) as image:
        image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (args.size, args.size),
                             method=Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].to(args.device, torch.float32)
    tensor = tensor.div(127.5).sub(1)
    model = load_vae(args.checkpoint, device=args.device)
    with torch.inference_mode():
        z = model.encode(tensor).mode()
        high = model.decode(z)
        low = model.decode_low(model.bottleneck_encode(z))
    args.output.mkdir(parents=True, exist_ok=True)
    image.save(args.output / "input.png")
    save_image(high, args.output / "high.png")
    save_image(low, args.output / "low.png")


if __name__ == "__main__":
    main()
