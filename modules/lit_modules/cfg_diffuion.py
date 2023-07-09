import typing

import pytorch_lightning as pl
import ema_pytorch
import torch
from torch import nn

import modules.diffusion
from modules.unet import UNet


class CFGDiffusion(pl.LightningModule):
    def __init__(
            self,
            num_classes: int,
            n_features_list: typing.List or typing.Tuple = (64, 128, 256),
            use_attention_list: typing.List or typing.Tuple = (False, True, True),
            embedding_dim: int = 256,
            num_steps: int = 1000,
            ema_beta=0.995,
            ema_update_after_step=100,
            ema_update_every=10,
            class_drop_prob: float = 0.1
    ):
        super().__init__()
        self.class_drop_prob = class_drop_prob
        self.num_classes = num_classes
        self.model = UNet(num_classes, n_features_list, use_attention_list, embedding_dim)
        self.ema_model = ema_pytorch.EMA(
            self.model, beta=ema_beta, update_after_step=ema_update_after_step, update_every=ema_update_every
        )
        self.diffusion = modules.diffusion.Diffusion(num_steps)

        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=3e-4)

    def training_step(self, batch, *args, **kwargs):
        images, classes = batch

        t = torch.randint(0, self.num_classes, size=(images.shape[0],)).to(images.device)
        noised_images, noise = self.diffusion.noise_images(images, t)

        predicted_noise = self.model.forward(noised_images, t, classes, class_drop_prob=self.class_drop_prob)

        loss = nn.functional.mse_loss(predicted_noise, noise)

        self.ema_model.update()
        self.log("train_loss", loss, prog_bar=True)
        return loss
