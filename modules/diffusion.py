import math

import torch
from torch import nn
from tqdm import trange


def cosine_beta_schedule(num_steps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).float()


class Diffusion(nn.Module):
    def __init__(self, num_steps: int = 1000):
        super().__init__()

        betas = cosine_beta_schedule(num_steps)
        alphas = 1 - betas
        alpha_hat = torch.cumprod(alphas, 0)
        self.num_steps = num_steps
        self.register_buffer("alpha_hat", alpha_hat)

    def noise_images(self, images, t: torch.Tensor):
        noise = torch.randn_like(images)
        t = self.alpha_hat[t][:, None, None, None]

        noised_images = images * torch.sqrt(t) + noise * torch.sqrt(1 - t)

        return noised_images, noise

    def sample(self, model, noises, classes):
        assert noises.shape[0] == classes.shape[0]
        betas = cosine_beta_schedule(self.num_steps).to(noises.device)
        alphas = 1 - betas
        alpha_hat = torch.cumprod(alphas, 0)

        with torch.no_grad():
            for i in trange(self.num_steps - 1, 0, -1):
                t = torch.full([classes.shape[0]], i, device=classes.device, dtype=torch.long)
                predicted_noise = model(noises, t, classes)

                epsilon = torch.randn_like(noises) if i > 1 else 0

                noises = 1 / torch.sqrt(alphas[i]) * (
                        noises - ((1 - alphas[i]) / torch.sqrt(1 - alpha_hat[i])) * predicted_noise
                ) + epsilon * torch.sqrt(betas[i])

        return noises








