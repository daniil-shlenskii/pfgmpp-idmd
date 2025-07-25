import torch

from pfgmpp_kernel import sample_noise


# TODO: make it more similar to sid implementation
def sid_sampler(
    net, *, latents, class_labels=None, init_sigma=2.5, D="inf", **sampling_kwargs
):
    z = sample_noise(latents=latents, sigma=init_sigma, D=D)
    x = net(z, (init_sigma*torch.ones(z.shape[0],1,1,1)).to(z.device), class_labels, **sampling_kwargs)
    return x

