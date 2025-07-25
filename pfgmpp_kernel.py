import numpy as np
import torch
from torch import Tensor
from torch.distributions import Beta

EPS = 1e-8


# TODO: not deterministic for given latents, which is important for generate.py reproducibility
def sample_noise(
    *,
    latents: Tensor,
    sigma: Tensor | float,
    #
    D: int | str = "inf",
):
    if isinstance(sigma, (float, int)):
        sigma = torch.tensor([sigma] * len(latents), device=latents.device, dtype=latents.dtype)

    if D == "inf": # EDM case
        noise = latents * sigma.view(-1, 1, 1, 1)
    else: # PFGMPP case
        assert isinstance(D, int)
        N = np.prod(latents.shape[1:])

        # Convert sigma to r
        r = sigma.reshape(-1) * D**0.5

        # Sample from inverse-beta distribution
        beta_gen = Beta(torch.FloatTensor([N / 2.]), torch.FloatTensor([D / 2.]))
        sample_norm = beta_gen.sample().to(latents.device).to(latents.dtype)
        sample_norm = torch.clamp(sample_norm, min=1e-3, max=1-1e-3)
        inverse_beta = sample_norm / (1 - sample_norm + EPS)

        # Sampling from p_r(R) by change-of-variable
        R = r * torch.sqrt(inverse_beta + EPS)
        R = R.view(len(sample_norm), -1)

        # Uniformly sample the angle component
        gaussian = latents.reshape(len(latents), -1)
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)

        # Construct the perturbation for x
        noise = unit_gaussian * R
        noise = noise.float().view_as(latents)
    return noise
