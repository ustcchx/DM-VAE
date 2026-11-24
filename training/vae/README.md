# Training the tokenizer

Trains the foundation-model-free DM-VAE (one encoder, high/low decoders with pixel + VGG-LPIPS loss, patch discriminator on the high branch plus manifold alignment, and 1×1 projection orthogonality losses). DINO-aligned variants require a separate teacher and feature-alignment loss.

## Setup

Run all commands from the repository root in the same environment:

```bash
python -m pip install torchvision lpips
```

Prepare the ImageNet-1K training split in `ImageFolder` layout. The loader resizes the shorter edge to 256, applies a random 256 × 256 crop and horizontal flip, and downloads pretrained VGG weights for LPIPS on first run.

## Train

```bash
torchrun --standalone --nproc-per-node=8 -m training.vae.train \
  --data /datasets/imagenet/train --output runs/dmvae-f16d32 \
  --high-channels 32 --low-channels 16 --batch-size 32 --epochs 50
```

- **f16d64:** Set `--high-channels 64 --low-channels 32` and use a fresh `--output` directory.
- **Single GPU:** Run with `python -m training.vae.train`.
- **Batch size:** `--batch-size` is per process (8 × 32 = 256 global batch size, matching the paper).
- **Offline sanity check:** Run on a small `ImageFolder` with `--epochs 1 --batch-size 1 --workers 0 --lpips-weight 0` (full training defaults to `--lpips-weight 1.0`).

## Defaults & checkpoints

- **Optimization:** Adam (`lr=1e-4`, `betas=(0.5, 0.9)`), low-branch weight `0.2`, KL weight `1e-6`, adaptive weights for manifold losses, and high-branch adversarial loss after `--disc-start`. Uses BF16 on supported CUDA GPUs (FP32 otherwise).
- **Resuming:** Checkpoints store model/EMA weights, discriminator, optimizers, adaptive weights, data position, and RNG states (crops and flips are deterministic per epoch and image index). Resume by adding `--resume /path/to/checkpoint.pt`, or pass a new `--output` directory to branch from an earlier checkpoint. Keep the dataset order, process count, batch size, and model/optimizer settings unchanged when resuming.
- **Inference:** `dmvae.load_vae` and the reconstruction script load the EMA weights by default.