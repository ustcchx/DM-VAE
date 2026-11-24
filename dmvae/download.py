"""Download a matched DM-VAE checkpoint from the versioned model archive."""

import argparse
from functools import lru_cache
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


@lru_cache(maxsize=None)
def archive_paths(repo, revision):
    """Look up the files in a pinned model revision by their LFS content digest."""
    from huggingface_hub import HfApi

    siblings = HfApi().model_info(repo, revision=revision, files_metadata=True).siblings
    paths = {}
    for file in sorted(siblings, key=lambda item: (item.rfilename.count("/"), item.rfilename)):
        if file.lfs:
            paths.setdefault(file.lfs.sha256, file.rfilename)
    return paths


def download(name, directory):
    entry = CATALOG[name]
    target = directory / entry["local_name"]
    if target.exists():
        if sha256(target) == entry["sha256"]:
            return target
        raise ValueError(f"Checksum mismatch for existing file {target}. Move it before retrying.")

    try:
        from huggingface_hub import constants, hf_hub_download
        from huggingface_hub.file_download import repo_folder_name
    except ImportError as error:
        raise SystemExit('Install the downloader with: python -m pip install -e ".[hub]"') from error

    repo, revision, digest = entry["repo"], entry["revision"], entry["sha256"]
    cache = Path(constants.HF_HUB_CACHE) / repo_folder_name(repo_id=repo, repo_type="model")
    source = cache / "blobs" / digest
    if not source.is_file():
        filename = archive_paths(repo, revision).get(digest)
        if filename is None:
            raise ValueError(f"Could not find {name} in the pinned revision of {repo}.")
        source = Path(hf_hub_download(repo_id=repo, revision=revision, filename=filename))
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
