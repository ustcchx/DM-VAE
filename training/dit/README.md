# Train the class-conditional generator

Trains a DiT velocity model in the frozen DM-VAE latent space with a TRA head (block 8) predicting the clean skeleton (16ch for `f16d32`, 32ch for `f16d64`). Run all commands from the repository root.

## 1. Extract features

```bash
python -m dmvae.download f16d32
python -m training.dit.extract \
  --images /path/to/imagenet/train \
  --vae weights/dmvae-f16d32.safetensors \
  --stats generation/configs/f16d32.json \
  --output features/f16d32
```

- **Data & disk:** Expects ImageNet in `ImageFolder` layout (`train/<synset>/<image>`). Caches ADM 256 center crops + flips (~117 GiB for `f16d32`, ~235 GiB for `f16d64`).
- **Custom VAE:** Omit `--stats` to estimate and save `stats.json` alongside the cache (pass it to `generation.sample --stats`).
- **Resume:** Re-run the same command to resume from completed shards.

## 2. Train `f16d32`

```bash
# 400 epochs (EMA 0.9999)
torchrun --standalone --nproc-per-node=8 -m training.dit.train \
  --cache features/f16d32/manifest.json \
  --output runs/dit-f16d32-400ep \
  --batch-size 64 --accum-steps 2 --epochs 400

# 64 epochs (EMA 0.9995)
torchrun --standalone --nproc-per-node=8 -m training.dit.train \
  --cache features/f16d32/manifest.json \
  --output runs/dit-f16d32-64ep \
  --batch-size 64 --accum-steps 2 --epochs 64 --ema-decay 0.9995

# Sample from EMA checkpoint
python -m generation.sample \
  --vae weights/dmvae-f16d32.safetensors \
  --dit runs/dit-f16d32-400ep/ema.pt \
  --stats features/f16d32/stats.json \
  --recipe 400ep-ig --classes 88 207 --output samples
```

- **Defaults:** Global batch 1,024 (`gpus × batch × accum`), `lr=2e-4`, Muon + AdamW, BF16. Use `--steps 1` for a quick test (LightningDiT and IG settings).
- **Checkpoints:** Saves `latest.pt` (add `--resume` with the same settings to continue) and `ema.pt` (for sampling).
- **Loss:** Velocity MSE + velocity cosine distance + `0.5×` TRA skeleton cosine distance (LightningDiT Setting).

## 3. Train `f16d64`

```bash
python -m dmvae.download f16d64
python -m training.dit.extract \
  --images /path/to/imagenet/train --vae weights/dmvae-f16d64.safetensors \
  --stats generation/configs/f16d64.json --output features/f16d64

torchrun --standalone --nproc-per-node=8 -m training.dit.train \
  --cache features/f16d64/manifest.json --vae weights/dmvae-f16d64.safetensors \
  --output runs/dit-f16d64-400ep --epochs 400 \
  --batch-size 64 --accum-steps 2 --ema-decay 0.9997

python -m generation.sample \
  --vae weights/dmvae-f16d64.safetensors --dit runs/dit-f16d64-400ep/ema.pt \
  --stats features/f16d64/stats.json --recipe d64-400ep-ig --classes 88 207
```

- Requires `--vae` during training and uses EMA decay `0.9997`.
- Uses shifted time $t = \sigma(\epsilon - 0.69)$ and skeleton-weighted velocity loss $Pv + (1-t)(I-P)v$ (via the frozen $64 \times 32$ decoder projection).

## References

Built on [LightningDiT](https://github.com/hustvl/LightningDiT), [Muon](https://github.com/KellerJordan/Muon), and [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion) preprocessing.