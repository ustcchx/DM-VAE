"""Regression checks for guidance in raw VAE coordinates and its inference cost."""

import unittest

import torch

from generation.sampling import Guidance, euler_sample, skeleton_guidance


class TwoChannelTokenizer:
    """Project onto channel zero so that channel one is entirely detail."""

    def bottleneck_encode(self, z):
        return z[:, :1]

    def bottleneck_decode(self, low):
        return torch.cat((low, torch.zeros_like(low)), dim=1)


class CountingFlow:
    num_classes = 1000

    def __init__(self):
        self.calls = []

    def __call__(self, z, t, labels, return_repa=False):
        self.calls.append((labels.item(), return_repa))
        velocity = torch.zeros_like(z)
        if return_repa:
            return velocity, torch.ones_like(z[:, :1])
        return velocity


class SkeletonGuidanceTest(unittest.TestCase):
    def test_skeleton_changes_but_detail_is_preserved(self):
        def tensor(values):
            return torch.tensor(values, dtype=torch.float32).reshape(1, 2, 1, 1)

        z_t = tensor([2, 1])
        velocity = tensor([0.5, 1.25])
        means, scales = tensor([3, -4]), tensor([2, 0.5])
        remaining = 0.5
        guided = skeleton_guidance(TwoChannelTokenizer(), z_t, velocity,
                                   -torch.ones_like(z_t[:, :1]), remaining, 2.0, means, scales)
        # In raw coordinates the skeleton is +4.125 and the norm-matched TRA
        # prediction is -4.125. Scale 2 adds their difference to the final skeleton.
        expected_raw_skeleton = 4.125 + (4.125 - (-4.125))
        expected = ((expected_raw_skeleton - 3) * 2 - 2) / remaining
        self.assertAlmostEqual(guided[0, 0, 0, 0].item(), expected, places=4)
        torch.testing.assert_close(guided[:, 1], velocity[:, 1])

    def test_only_cfg_adds_a_second_forward_pass(self):
        model = CountingFlow()
        z = torch.zeros(1, 2, 1, 1)
        euler_sample(model, TwoChannelTokenizer(), z, torch.tensor([42]), z, torch.ones_like(z),
                     Guidance(1.5, 0, 0.5), Guidance(1.25, 0.25, 0.75), steps=4)
        self.assertEqual(model.calls, [(42, True), (42, True), (1000, False),
                                       (42, False), (1000, False), (42, False)])


if __name__ == "__main__":
    unittest.main()
