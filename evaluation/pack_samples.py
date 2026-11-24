"""Pack class-balanced ImageNet PNGs from generation.sample into an ADM sample batch."""

import argparse
import os
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

FILENAME = re.compile(r"(\d+)_class(\d+)\.png")


def pack(directory, output, count):
    if count < 1:
        raise ValueError("Sample count must be positive.")
    files = {}
    for path in directory.glob("*.png"):
        if path.name.startswith("._"):
            continue
        match = FILENAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected sample filename: {path.name}")
        index, label = map(int, match.groups())
        if index >= count or index in files or label != index % 1000:
            raise ValueError(f"Unexpected sample index or ImageNet label: {path.name}")
        files[index] = path
    if len(files) != count:
        raise ValueError(f"Expected {count} unique images, found {len(files)}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        archive = Path(temp) / "samples.npz"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as batch:
            with batch.open("arr_0.npy", "w", force_zip64=True) as array:
                np.lib.format.write_array_header_1_0(array, {
                    "descr": "|u1", "fortran_order": False, "shape": (count, 256, 256, 3)
                })
                for index in range(count):
                    with Image.open(files[index]) as image:
                        if image.mode != "RGB" or image.size != (256, 256):
                            raise ValueError(f"Expected 256 × 256 RGB: {files[index]}")
                        array.write(image.tobytes())
        os.replace(archive, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50000)
    args = parser.parse_args()
    pack(args.images, args.output, args.count)
    print(f"Packed {args.count} images in {args.output}")


if __name__ == "__main__":
    main()
