"""Read paired posterior samples and clean skeleton targets from safetensors shards."""
import bisect
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.data import Dataset


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PairedLatents(Dataset):
    def __init__(self, path, verify=False):
        path = Path(path)
        self.manifest = data = json.loads(path.read_text())
        if (data["schema"] != "dflow_paired_latent_cache/v1" or data["state"] != "complete"
                or data["posterior"] != "sample" or data["repa_target"] != "bottleneck_encode(posterior.mode())"
                or (data["latent_shape"], data["target_shape"]) not in
                   (([32, 16, 16], [16, 16, 16]), ([64, 16, 16], [32, 16, 16]))):
            raise ValueError("Expected a complete paired cache for f16d32/low16 or f16d64/low32.")
        self.high = data["latent_shape"][0]
        self.low = data["target_shape"][0]
        stats = data["stats"]
        self.mean = torch.tensor(stats["channel_means"], dtype=torch.float32).reshape(self.high, 1, 1)
        self.std = torch.tensor(stats["channel_std"], dtype=torch.float32).reshape(self.high, 1, 1)
        if not (self.mean.isfinite().all() and self.std.isfinite().all() and (self.std > 0).all()):
            raise ValueError("Invalid channel statistics.")
        self.ends = []
        self.paths = []
        count = 0
        for shard in data["shards"]:
            name = shard["file"]
            if Path(name).name != name or shard["count"] <= 0 or shard["start_index"] != count:
                raise ValueError(f"Invalid shard entry: {name}")
            shard_path = path.parent / name
            if shard_path.stat().st_size != shard["bytes"]:
                raise ValueError(f"Incomplete shard: {name}")
            if verify and ("sha256" not in shard or file_digest(shard_path) != shard["sha256"]):
                raise ValueError(f"Missing or mismatched SHA-256 for {name}.")
            with safe_open(str(shard_path), framework="pt") as reader:
                for field in ("latents", "latents_flip", "low_targets", "low_targets_flip"):
                    channels = self.low if field.startswith("low") else self.high
                    tensor = reader.get_slice(field)
                    if tensor.get_shape() != [shard["count"], channels, 16, 16] or tensor.get_dtype() != "F32":
                        raise ValueError(f"Invalid {field} in {name}.")
                for field in ("labels", "indices"):
                    tensor = reader.get_slice(field)
                    if tensor.get_shape() != [shard["count"]] or tensor.get_dtype() not in ("I32", "I64"):
                        raise ValueError(f"Invalid {field} in {name}.")
            count += shard["count"]
            self.ends.append(count)
            self.paths.append(shard_path)
        if count != data["count"] or count == 0:
            raise ValueError("Incorrect number of cached images.")
        self._reader = self._key = None

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, key):
        index, flipped = key
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self.ends, index)
        row = index - (self.ends[shard - 1] if shard else 0)
        reader_key = (os.getpid(), shard)
        if self._key != reader_key:
            self._reader = safe_open(str(self.paths[shard]), framework="pt")
            self._key = reader_key
        if self._reader.get_slice("indices")[row].item() != index:
            raise ValueError("Cached row has an incorrect sample index.")
        suffix = "_flip" if flipped else ""
        latent = self._reader.get_slice("latents" + suffix)[row]
        target = self._reader.get_slice("low_targets" + suffix)[row]
        label = self._reader.get_slice("labels")[row]
        return (latent - self.mean) / self.std, label.long(), target

    def __getstate__(self):
        return {**self.__dict__, "_reader": None, "_key": None}
