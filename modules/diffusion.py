import math

import torch
from torch import nn
from tqdm import trange, tqdm


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
    def __init__(self, num_steps: int = 1000, num_sample_steps: int = 250, ddim_sampling_eta=1.,):
        super().__init__()

        self.num_sample_steps = num_sample_steps
        self.ddim_sampling_eta = ddim_sampling_eta

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

    @torch.no_grad()
    def sample(self, model, x0, classes, cond_scale=6., rescaled_phi=0.7, return_history: bool = False):
        assert x0.shape[0] == classes.shape[0]
        model = model.eval()

        times = torch.linspace(-1, self.num_steps - 1,
                               steps=self.num_sample_steps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        sqrt_recip_alphas_cumprod = torch.sqrt(1 / self.alpha_hat)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alpha_hat - 1)
        history = [x0]
        xt = x0

        for time, time_next in tqdm(time_pairs, desc="Predicting images"):
            current_time = torch.full((x0.shape[0],), time, dtype=torch.long, device=x0.device)
            predicted_noise = model.forward_with_cond_scale(xt, current_time, classes, cond_scale=cond_scale, rescaled_phi=rescaled_phi)
            predicted_x_start = xt * sqrt_recip_alphas_cumprod[time] - predicted_noise * sqrt_recipm1_alphas_cumprod[time]
            predicted_x_start = torch.clip(predicted_x_start, -1, 1)

            if time_next < 0:
                xt = predicted_x_start
                history.append(xt)
                continue

            alpha = self.alpha_hat[time]
            alpha_next = self.alpha_hat[time_next]

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(xt)

            xt = predicted_x_start * alpha_next.sqrt() + c * predicted_noise + sigma * noise
            history.append(xt)

        if return_history:
            return torch.cat([i[None] for i in history], dim=0) * 0.5 + 0.5
        return xt * 0.5 + 0.5














