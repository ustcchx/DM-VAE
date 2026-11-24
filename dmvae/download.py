"""Download matched DM-VAE checkpoints from the public Hugging Face repository."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

CATALOG = json.loads(Path(__file__).with_name("weights.json").read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name, directory):
    entry = CATALOG[name]
    target = directory / entry["local_name"]
    if target.exists():
        if sha256(target) == entry["sha256"]:
            return target
        raise ValueError(f"Checksum mismatch for existing file {target}. Move it before retrying.")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit('Install the downloader with: python -m pip install -e ".[hub]"') from error

    source = Path(hf_hub_download(repo_id=entry["repo"], revision=entry["revision"], filename=entry["filename"]))
    if sha256(source) != entry["sha256"]:
        raise ValueError(f"Downloaded file failed SHA-256 verification for {name}.")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as staging:
        ready = Path(staging) / target.name
        try:
            os.link(source.resolve(), ready)
        except OSError:
            shutil.copyfile(source, ready)
        os.replace(ready, target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=sorted(CATALOG))
    parser.add_argument("--output", type=Path, default=Path("weights"))
    args = parser.parse_args()
    for model in args.models:
        print(download(model, args.output))


if __name__ == "__main__":
    main()
