"""Train the foundation-model-free tokenizer on an ImageFolder training split."""
import argparse
import copy
import fcntl
import hashlib
import itertools
import math
import os
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from dmvae import DMVAE


class TrainingImages(datasets.ImageFolder):
    """Anchor each augmentation to its image and epoch, independent of loader workers."""
    def __init__(self, root, transform, seed):
        super().__init__(root, transform)
        self.seed = seed
        self.epoch = 0

    def __getitem__(self, index):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed + self.epoch * len(self) + index)
            return super().__getitem__(index)


class TrainingSampler(DistributedSampler):
    skip = 0

    def __iter__(self):
        return itertools.islice(super().__iter__(), self.skip, None)

    def __len__(self):
        return super().__len__() - self.skip


@contextmanager
def reserve_output(args, device, rank, world):
    """Keep concurrent runs and stale resumes from overwriting an existing checkpoint."""
    lock = None
    error = [None]
    try:
        if rank == 0:
            try:
                args.output.mkdir(parents=True, exist_ok=True)
                lock = (args.output / ".train.lock").open("a")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                checkpoints = sorted(
                    (p for p in args.output.glob("step_*.pt") if p.stem[5:].isdigit()),
                    key=lambda p: int(p.stem[5:]),
                )
                if checkpoints and (not args.resume or checkpoints[-1].resolve() != args.resume.resolve()):
                    raise ValueError(f"Resume from {checkpoints[-1]} or choose a new output directory.")
            except (OSError, ValueError) as exc:
                error[0] = str(exc)
        if world > 1:
            dist.broadcast_object_list(error, src=0, device=device)
        if error[0]:
            raise RuntimeError(f"Cannot use output directory {args.output}: {error[0]}")
        yield
    finally:
        if lock is not None:
            lock.close()


class PatchDiscriminator(nn.Sequential):
    """PatchGAN with three strided layers for the high-dimensional reconstruction."""
    def __init__(self):
        layers = [nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        for c_in, c_out, stride in ((64, 128, 2), (128, 256, 2), (256, 512, 1)):
            layers.extend([nn.Conv2d(c_in, c_out, 4, stride, 1, bias=False),
                           nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2, inplace=True)])
        layers.append(nn.Conv2d(512, 1, 4, 1, 1))
        super().__init__(*layers)
        for layer in self:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, 0.0, 0.02)
            if isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight, 1.0, 0.02)
                nn.init.zeros_(layer.bias)


    def forward(self, images, reconstruction=None):
        logits = super().forward(images)
        if reconstruction is None:
            return logits
        return logits, super().forward(reconstruction)


def rank_penalty(residual):
    batch, channels, height, width = residual.shape
    x = residual.reshape(batch, channels, -1)
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / (x.square().mean((1, 2), keepdim=True) + 1e-4).sqrt()
    x = x / math.sqrt(height * width)
    cov = x @ x.transpose(1, 2)
    trace = cov.diagonal(dim1=1, dim2=2).sum(-1)
    return (1 - cov.square().sum((1, 2)) / (trace.square() + 1e-4)).clamp(0, 1).mean()


def orthogonality_penalty(model):
    down = model.bottleneck_encode.weight[:, :, 0, 0].float()
    up = model.bottleneck_decode.weight[:, :, 0, 0].float()
    identity = torch.eye(down.shape[0], device=down.device)
    return (F.mse_loss(down @ down.T, identity) + F.mse_loss(up.T @ up, identity)
            + 0.5 * F.mse_loss(down, up.T))


def adaptive_weight(reference, auxiliary, weight, maximum):
    reference_grad = torch.autograd.grad(reference, weight, retain_graph=True)[0]
    auxiliary_grad = torch.autograd.grad(auxiliary, weight, retain_graph=True)[0]
    ratio = reference_grad.float().norm() / (auxiliary_grad.float().norm() + 1e-2)
    return ratio.detach().clamp(0, maximum)


def reconstruction_loss(pred, images, perceptual, perceptual_weight):
    error = (pred.float() - images).abs()
    if perceptual is not None:
        error = error + perceptual_weight * perceptual(pred.float(), images)
    return error.flatten(1).sum(1).mean()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="ImageFolder root containing class folders")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--high-channels", type=int, default=32)
    p.add_argument("--low-channels", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32, help="Images per process")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lpips-weight", type=float, default=1.0)
    p.add_argument("--disc-start", type=int, default=1001, help="First generator step with adversarial loss")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--max-steps", type=int, help="Optional total optimizer-step limit")
    p.add_argument("--resume", type=Path, help="Training checkpoint saved by this command")
    p.add_argument("--seed", type=int, default=42)
    return p


def train(args, device, rank, world):
    torch.manual_seed(args.seed)
    transform = transforms.Compose([transforms.Resize(args.image_size),
                                    transforms.RandomCrop(args.image_size),
                                    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                    transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    dataset = TrainingImages(args.data, transform, args.seed)
    sampler = TrainingSampler(dataset, num_replicas=world, rank=rank, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        generator=torch.Generator().manual_seed(args.seed),
                        num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True)
    epoch_batches = len(loader)
    if epoch_batches == 0:
        raise ValueError("Each process needs at least one full batch. Reduce --batch-size.")
    model = DMVAE(args.high_channels, args.low_channels).to(device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    discriminator = PatchDiscriminator().to(device)
    opt_g = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    next_step = start_epoch = start_batch = 0
    manifold_weight = torch.tensor(1.0, device=device)
    rank_weight = torch.tensor(1.0, device=device)
    digest = hashlib.sha256()
    root = str(dataset.root).rstrip(os.sep) + os.sep
    for filename, label in dataset.samples:
        digest.update(os.fsencode(filename.removeprefix(root)) + b"\0" + str(label).encode() + b"\n")
    contract = dict(index_sha256=digest.hexdigest(), batches=epoch_batches, world=world,
                    batch_size=args.batch_size, image_size=args.image_size, seed=args.seed,
                    channels=(args.high_channels, args.low_channels), learning_rate=args.learning_rate,
                    disc_start=args.disc_start, lpips_weight=args.lpips_weight, device=device.type)
    rng = None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if "contract" in saved and saved["contract"] != contract:
            raise ValueError("Resume requires matching data, model, batch and training settings.")
        if "rng" in saved:
            rng = saved["rng"][rank]
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        discriminator.load_state_dict(saved["discriminator"])
        opt_g.load_state_dict(saved["opt_g"])
        opt_d.load_state_dict(saved["opt_d"])
        next_step, start_epoch, start_batch = saved["next_step"], saved["epoch"], saved["batch"]
        manifold_weight, rank_weight = saved["aux_weights"].to(device).unbind()
        del saved
    network, critic = model, discriminator
    if world > 1:
        device_ids = [device.index] if device.type == "cuda" else None
        network = DistributedDataParallel(model, device_ids=device_ids)
        critic = DistributedDataParallel(discriminator, device_ids=device_ids)
    perceptual = None
    if args.lpips_weight:
        import lpips
        perceptual = lpips.LPIPS(net="vgg").eval().requires_grad_(False).to(device)
    total_steps = min(args.epochs * epoch_batches, args.max_steps or args.epochs * epoch_batches)
    if next_step > total_steps or (start_epoch, start_batch) != divmod(next_step, epoch_batches):
        raise ValueError("Saved data position does not fit this training budget or split.")
    torch.manual_seed(args.seed + rank)
    if rng is not None:
        torch.set_rng_state(rng[0])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng[1], device)
    elif args.resume and rank == 0:
        print("Legacy checkpoint has no RNG state. Continuing from its saved weights and data position.", flush=True)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        dataset.epoch = epoch
        offset = start_batch if epoch == start_epoch else 0
        sampler.skip = offset * args.batch_size
        for batch, (images, _) in enumerate(loader, start=offset):
            if next_step >= total_steps:
                return
            images = images.to(device, non_blocking=True)
            opt_g.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                high, low, mu, logvar, mu_hat = network(images)
            mu, mu_hat, logvar = mu.float(), mu_hat.float(), logvar.float()
            nll = reconstruction_loss(high, images, perceptual, args.lpips_weight)
            low_nll = reconstruction_loss(low, images, perceptual, args.lpips_weight)
            kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).flatten(1).sum(1).mean()
            residual_projection = mu_hat - mu.detach()
            residual_encoder = mu - mu_hat.detach()
            manifold_e = residual_encoder.square().mean()
            rank_e = rank_penalty(residual_encoder)
            manifold = 0.5 * (residual_projection.square().mean() + manifold_e)
            rank_loss = 0.5 * (rank_penalty(residual_projection) + rank_e)
            if next_step % 20 == 0:
                weights = torch.stack([adaptive_weight(nll, term, model.quant_conv.weight, 1e5)
                                       for term in (manifold_e, rank_e)])
                if world > 1:
                    dist.all_reduce(weights)
                    weights /= world
                manifold_weight = 0.8 * manifold_weight + 0.2 * weights[0]
                rank_weight = 0.8 * rank_weight + 0.2 * weights[1]
            # We actually drop the rank_loss in training
            loss = (nll + 0.2 * low_nll + 1e-6 * kl + 0.1 * manifold_weight * manifold
                    + 0.000 * rank_weight * rank_loss + 10 * orthogonality_penalty(model))
            adversarial = next_step >= args.disc_start
            if adversarial:
                discriminator.requires_grad_(False)
                fake = -discriminator(high.float()).mean()
                weight = 0.5 * adaptive_weight(nll, fake, model.decoder.conv_out.weight, 1e4)
                loss = loss + weight * fake
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt_g.step()
            if adversarial:
                discriminator.requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)
                real_logits, fake_logits = critic(images, high.detach().float())
                disc_loss = 0.5 * (F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean())
                disc_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0, error_if_nonfinite=True)
                opt_d.step()
            next_step += 1
            with torch.no_grad():
                decay = min(0.9999, (1 + next_step) / (10 + next_step))
                for shadow, current in zip(ema.parameters(), model.parameters()):
                    shadow.lerp_(current, 1 - decay)
            if rank == 0 and (next_step == 1 or next_step % 20 == 0):
                print(f"step={next_step} high_nll={nll.item():.3f} low_nll={low_nll.item():.3f}"
                      f" manifold={manifold.item():.5f} kl={kl.item():.3f}", flush=True)
            should_save = next_step % args.save_every == 0 or next_step == total_steps
            if should_save:
                rank_rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
                rng_states = [rank_rng]
                if world > 1:
                    rng_states = [None] * world if rank == 0 else None
                    dist.gather_object(rank_rng, rng_states, dst=0)
            if rank == 0 and should_save:
                finished_epoch = batch + 1 == epoch_batches
                state = dict(model=model.state_dict(), ema=ema.state_dict(), discriminator=discriminator.state_dict(),
                             opt_g=opt_g.state_dict(), opt_d=opt_d.state_dict(), next_step=next_step,
                             contract=contract, rng=rng_states,
                             epoch=epoch + int(finished_epoch), batch=0 if finished_epoch else batch + 1,
                             aux_weights=torch.stack([manifold_weight, rank_weight]))
                path = args.output / f"step_{next_step:07d}.pt"
                temporary = path.with_suffix(".tmp")
                torch.save(state, temporary)
                os.replace(temporary, path)
            if next_step >= total_steps:
                return


def main():
    p = parser()
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.save_every, args.high_channels, args.low_channels) < 1:
        p.error("Epochs, batch size, checkpoint interval and channel counts must be positive.")
    if args.learning_rate <= 0 or args.disc_start < 0:
        p.error("The learning rate must be positive and --disc-start must be nonnegative.")
    if args.low_channels > args.high_channels or args.workers < 0 or args.lpips_weight < 0:
        p.error("Invalid channels, worker count or perceptual weight.")
    if args.image_size < 32 or args.image_size % 16:
        p.error("--image-size must be at least 32 and divisible by 16")
    if args.max_steps is not None and args.max_steps < 1:
        p.error("--max-steps must be positive")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        with reserve_output(args, device, rank, world):
            train(args, device, rank, world)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
