"""Train the class-conditional DM-VAE generator with topological representation alignment."""
import argparse
from copy import deepcopy
from contextlib import nullcontext
import fcntl
import itertools
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from .cache import PairedLatents, file_digest
from .objective import flow_inputs, loss_terms, normalized_skeleton_projector
from .optimizer import make_optimizers


class OrientedSampler(Sampler):
    """Partition a single shuffled epoch and select the cached orientation per image."""
    def __init__(self, dataset, world, rank, seed, epoch, skip):
        self.base = DistributedSampler(dataset, world, rank, shuffle=True, seed=seed, drop_last=True)
        self.base.set_epoch(epoch)
        self.flips = np.random.RandomState(seed + epoch).uniform(size=len(dataset)) <= 0.5
        self.skip = skip

    def __iter__(self):
        for index in itertools.islice(self.base, self.skip, None):
            yield index, bool(self.flips[index])

    def __len__(self):
        return len(self.base) - self.skip


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.mul_(decay).add_(source, alpha=1 - decay)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def run(args, model_factory=None):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise ValueError("The CUDA training recipe requires BF16 support.")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    lock = None
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        available = torch.ones((), device=device, dtype=torch.int32)
        if rank == 0:
            lock = (args.output / ".train.lock").open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                available.zero_()
        if world > 1:
            dist.broadcast(available, src=0)
        if not available.item():
            raise RuntimeError("Another trainer is already using this output directory.")
        train(args, model_factory, rank, world, device)
    finally:
        if lock is not None:
            lock.close()
        if world > 1:
            dist.destroy_process_group()


def train(args, model_factory, rank, world, device):
    accumulation = args.accum_steps
    if accumulation < 1:
        raise ValueError("--accum-steps must be positive.")
    per_rank = args.batch_size
    if per_rank is None:
        if 1024 % (world * accumulation):
            raise ValueError("Set --batch-size when 1024 is not divisible by world size times accumulation.")
        per_rank = 1024 // (world * accumulation)
    if (min(per_rank, args.save_every, args.log_every) <= 0 or args.workers < 0
            or (args.ema_decay is not None and not 0 < args.ema_decay < 1) or args.lr <= 0):
        raise ValueError("Invalid training hyperparameters.")
    data = PairedLatents(args.cache, verify=args.verify_cache and rank == 0)
    wide = data.high == 64
    if wide and args.vae is None:
        raise ValueError("For f16d64, pass --vae with the tokenizer that generated this cache.")
    if args.vae is not None and file_digest(args.vae) != data.manifest["vae_ckpt_sha256"]:
        raise ValueError("The VAE checkpoint does not match the feature cache.")
    projector = None
    if wide:
        from dmvae import load_vae
        vae = load_vae(args.vae)
        projection = vae.bottleneck_decode
        if projection.bias is not None:
            raise ValueError("The f16d64 training objective requires a bias-free VAE projection.")
        projector = normalized_skeleton_projector(projection.weight, data.std).to(device)
        del vae, projection
    if model_factory is None:
        from generation.model import LatentDiT
        from generation.model_d64 import LatentDiT64
        model_factory = LatentDiT64 if wide else LatentDiT
    ema_decay = args.ema_decay if args.ema_decay is not None else (0.9997 if wide else 0.9999)
    global_batch = per_rank * world * accumulation
    updates_per_epoch = len(data) // global_batch
    if not updates_per_epoch:
        raise ValueError("The cache must contain at least one full global batch. Try --batch-size 1.")
    target_steps = args.steps if args.steps is not None else args.epochs * updates_per_epoch
    if target_steps < 1:
        raise ValueError("Training duration must be positive.")
    state_file = args.output / "latest.pt"
    if state_file.exists() and not args.resume:
        raise FileExistsError("Choose a new output directory or use --resume.")
    torch.manual_seed(args.seed)
    model = model_factory().to(device).train()
    forward = model
    if world > 1:
        forward = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    ema = deepcopy(model).eval().requires_grad_(False) if rank == 0 else None
    muon, adamw = make_optimizers(model, args.lr)
    contract = dict(cache=file_digest(args.cache), per_rank=per_rank, world=world, accumulation=accumulation, seed=args.seed,
                    ema_decay=ema_decay, lr=args.lr, device=device.type)
    torch.manual_seed(args.seed + rank)
    step = 0
    if args.resume:
        state = torch.load(state_file, map_location="cpu", weights_only=True)
        if state["contract"] != contract:
            raise ValueError("Resume requires the original cache and training settings.")
        model.load_state_dict(state["model"])
        if ema is not None:
            ema.load_state_dict(state["ema"])
        muon.load_state_dict(state["muon"])
        adamw.load_state_dict(state["adamw"])
        step = state["step"]
        cpu_rng, device_rng = state["rng"][rank]
        torch.set_rng_state(cpu_rng)
        if device.type == "cuda":
            torch.cuda.set_rng_state(device_rng, device)
        del state
    if step > target_steps:
        raise ValueError("The saved step exceeds the requested training duration.")
    if step == target_steps and rank == 0:
        atomic_save(ema.state_dict(), args.output / "ema.pt")
    if rank == 0:
        print(f"Training f16d{data.high} on {len(data)} images, global batch {global_batch}, "
              f"{updates_per_epoch} updates/epoch, starting at update {step} of {target_steps}", flush=True)

    while step < target_steps:
        epoch, offset = divmod(step, updates_per_epoch)
        sampler = OrientedSampler(data, world, rank, args.seed, epoch, offset * per_rank * accumulation)
        loader = DataLoader(data, batch_size=per_rank, sampler=sampler, num_workers=args.workers,
                            pin_memory=device.type == "cuda", drop_last=True,
                            generator=torch.Generator().manual_seed(args.seed + epoch * world + rank))
        metrics = torch.zeros(4, device=device)
        remaining = (updates_per_epoch - offset) * accumulation
        for microstep, (latent, labels, skeleton) in enumerate(itertools.islice(loader, remaining), start=1):
            update = microstep % accumulation == 0
            sync = forward.no_sync() if world > 1 and not update else nullcontext()
            latent = latent.to(device, dtype=torch.float32, non_blocking=True)
            labels, skeleton = labels.to(device, non_blocking=True), skeleton.to(device, non_blocking=True)
            noisy, time, velocity = flow_inputs(latent, logit_mean=-0.69 if wide else 0.0)
            with sync:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    pred, pred_low = forward(noisy, time, labels, return_repa=True)
                loss, mse, direction, alignment = loss_terms(pred, velocity, pred_low, skeleton, time, projector)
                (loss / accumulation).backward()
            metrics += torch.stack([loss, mse, direction, alignment]).detach() / accumulation
            if not update:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            muon.step()
            adamw.step()
            model.zero_grad(set_to_none=True)
            if ema is not None:
                update_ema(ema, model, ema_decay)
            step += 1
            if step == 1 or step % args.log_every == 0:
                if world > 1:
                    dist.all_reduce(metrics)
                    metrics /= world
                if rank == 0:
                    total, mse_value, cosine_value, tra_value = metrics.tolist()
                    print(f"step {step} loss {total:.4f} mse {mse_value:.4f} "
                          f"direction {cosine_value:.4f} TRA {tra_value:.4f}", flush=True)
            metrics.zero_()
            if step % args.save_every == 0 or step == target_steps:
                rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
                states = [rng]
                if world > 1:
                    states = [None] * world if rank == 0 else None
                    dist.gather_object(rng, states, dst=0)
                if rank == 0:
                    atomic_save(dict(step=step, contract=contract, model=model.state_dict(), ema=ema.state_dict(),
                                     muon=muon.state_dict(), adamw=adamw.state_dict(), rng=states), state_file)
                    atomic_save(ema.state_dict(), args.output / "ema.pt")
                if world > 1:
                    dist.barrier()
            if step == target_steps:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="Paired cache manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vae", type=Path, help="Frozen tokenizer used to write the cache (required for f16d64)")
    parser.add_argument("--batch-size", type=int, help="Images per process (default global batch 1024)")
    parser.add_argument("--accum-steps", type=int, default=1, help="Microbatches per optimizer update")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--epochs", type=int, default=400, help="Total complete training epochs (default: 400)")
    duration.add_argument("--steps", type=int, help="Total optimizer updates, including updates before --resume")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema-decay", type=float, help="Defaults to the selected 400-epoch EMA for each model")
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--verify-cache", action="store_true", help="Check every feature shard SHA-256 before training")
    parser.add_argument("--resume", action="store_true", help="Resume from OUTPUT/latest.pt")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
