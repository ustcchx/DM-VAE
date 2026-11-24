"""Hidden-matrix Muon with the auxiliary AdamW recipe used by DM-VAE."""
import math

import torch


def orthogonalize(gradient):
    matrix = gradient.to(torch.bfloat16)
    transpose = matrix.shape[-2] > matrix.shape[-1]
    if transpose:
        matrix = matrix.mT
    matrix = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        gram = matrix @ matrix.mT
        matrix = 3.4445 * matrix + (-4.775 * gram + 2.0315 * (gram @ gram)) @ matrix
    return matrix.mT if transpose else matrix


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            by_shape = {}
            for parameter in group["params"]:
                if parameter.grad is not None:
                    by_shape.setdefault(parameter.shape, []).append(parameter)
            for shape, parameters in by_shape.items():
                alpha = -group["lr"] * 0.2 * math.sqrt(max(shape))
                for start in range(0, len(parameters), 8):
                    chunk = parameters[start:start + 8]
                    gradients = []
                    for parameter in chunk:
                        state = self.state[parameter]
                        if "momentum" not in state:
                            state["momentum"] = torch.zeros_like(parameter)
                        buffer = state["momentum"].mul_(0.95).add_(parameter.grad)
                        gradients.append(parameter.grad.add(buffer, alpha=0.95))
                    updates = orthogonalize(torch.stack(gradients))
                    for parameter, update in zip(chunk, updates):
                        parameter.add_(update, alpha=alpha)


def make_optimizers(model, lr):
    hidden, auxiliary = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        use_muon = parameter.ndim == 2 and all(
            key not in name for key in ("embedding_table", "final_layer", "x_embedder")
        )
        (hidden if use_muon else auxiliary).append(parameter)
    return Muon(hidden, lr), torch.optim.AdamW(
        auxiliary, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
        fused=auxiliary[0].is_cuda,
    )
