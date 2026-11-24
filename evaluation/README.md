# ImageNet-256 generation evaluation

The generation table reports FID between 50,000 class-balanced samples and the ADM ImageNet-256 reference distribution. Use the official TensorFlow evaluator for the reported FID, sFID, Inception Score, precision and recall. Run it in its own environment with its upstream dependencies. [ADM evaluation code and setup](https://github.com/openai/guided-diffusion/tree/main/evaluations) · [ImageNet-256 reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz)

## Generate and score

Run these commands from the repository root after downloading the matching checkpoints. Use an empty image directory. Omitting `--classes` assigns 50 images to each of the 1,000 ImageNet classes. The published f16d32 protocol uses CUDA, bfloat16, 250 Euler updates and seed 1234.

```bash
python -m generation.sample \
  --dit weights/dit-f16d32-400ep.pt --vae weights/dmvae-f16d32.pt \
  --recipe 400ep-ig-cfg --device cuda --precision bf16 \
  --count 50000 --batch-size 8 --output outputs/fid400/png

python -m evaluation.pack_samples \
  --images outputs/fid400/png --output outputs/fid400/samples.npz

# In the ADM evaluator environment
python /path/to/guided-diffusion/evaluations/evaluator.py \
  /path/to/VIRTUAL_imagenet256_labeled.npz outputs/fid400/samples.npz
```

For multiple GPUs, run independent sampling processes on disjoint `--start-index` and `--count` ranges, writing to the same image directory. For example, four processes can each generate 12,500 images starting at 0, 12500, 25000 and 37500. The global sample index determines the class and seed. Pack once after all processes finish. The packer validates the entire index and label sequence and writes a uint8 NHWC array incrementally. The full NPZ needs 9.83 GB in addition to the PNGs.

For the 400-epoch f16d32 model, the reference run with this recipe reports FID 1.1711, sFID 4.0183 and IS 282.96. With `400ep-ig` the reference run reports FID 1.2478, sFID 4.1602 and IS 250.75. These are historical full-set results. Numerical differences can arise with different accelerators, attention kernels or precision. [Paper results](https://ustcchx.github.io/DM-VAE-page/#results)

## Reconstruction results

The tokenizer table compares reconstructions with their source images. The paper uses the ImageNet-1K validation split (50,000 images) and Anime-Faces-256 (5,000 images). PSNR is a paired pixel metric. rFID compares the two distributions of real and reconstructed images for the same split. The ADM generation command above uses a separate reference distribution.

The following command computes mean per-image PSNR and saves matching uint8 PNG pairs. It takes the deterministic posterior mean and scores the full decoder. Images that are already 256 × 256 keep their geometry. For other shapes the exporter uses the same bicubic center crop as the quick start. Supply the exact validation crops used by your comparison when reporting scores.

```bash
python -m evaluation.reconstruct --vae weights/dmvae-f16d32.pt \
  --images /path/to/validation --output outputs/recon-val

python -m pip install pytorch-fid
python -m pytorch_fid outputs/recon-val/reference outputs/recon-val/reconstructed
```

The second command computes rFID with the PyTorch FID Inception pool-3 features. The input crops, posterior choice and Inception implementation should be recorded with any new score. [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
