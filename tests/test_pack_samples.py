"""The ADM reader expects a contiguous uint8 NHWC array with ascending sample IDs."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from evaluation.pack_samples import pack


class ADMSampleTest(unittest.TestCase):
    def test_order_and_label_validation(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            for index in (2, 0, 1):
                Image.new("RGB", (256, 256), (index, 0, 0)).save(
                    directory / f"{index:06d}_class{index:04d}.png"
                )
            (directory / "._000000_class0000.png").write_bytes(b"resource fork")
            output = directory / "samples.npz"
            pack(directory, output, 3)
            with np.load(output) as batch:
                pixels = batch["arr_0"]
                self.assertEqual(pixels.shape, (3, 256, 256, 3))
                self.assertEqual(pixels.dtype, np.uint8)
                self.assertEqual(pixels[:, 0, 0, 0].tolist(), [0, 1, 2])
            (directory / "000002_class0002.png").rename(directory / "000002_class0004.png")
            with self.assertRaisesRegex(ValueError, "label"):
                pack(directory, output, 3)


if __name__ == "__main__":
    unittest.main()
