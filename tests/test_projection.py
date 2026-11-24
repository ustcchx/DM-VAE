"""The f16d64 projector lives in normalized coordinates and damps only detail."""

import unittest

import torch

from training.dit.objective import loss_terms, normalized_skeleton_projector


class ProjectedFlowTest(unittest.TestCase):
    def test_projection_uses_normalized_decoder_basis(self):
        weights = torch.cat((torch.eye(32), torch.eye(32))).view(64, 32, 1, 1)
        std = torch.cat((torch.ones(32), torch.full((32,), 2.0)))
        projection = normalized_skeleton_projector(weights, std)
        basis = weights[:, :, 0, 0] / std[:, None]
        torch.testing.assert_close(projection @ basis, basis)
        torch.testing.assert_close(projection @ projection, projection)
        torch.testing.assert_close(projection.trace(), torch.tensor(32.0))

    def test_detail_error_is_damped_as_data_time_approaches(self):
        projector = torch.diag(torch.tensor([1.0, 0.0]))
        prediction = torch.tensor([1.0, 3.0]).reshape(1, 2, 1, 1)
        target = torch.tensor([1.0, 1.0]).reshape_as(prediction)
        skeleton = torch.ones(1, 1, 1, 1)
        for time, expected_mse in [(0.0, 2.0), (0.5, 0.5), (1.0, 0.0)]:
            _, mse, _, tra = loss_terms(prediction, target, skeleton, skeleton,
                                       torch.tensor([time]), projector)
            self.assertAlmostEqual(mse.item(), expected_mse)
            self.assertAlmostEqual(tra.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
