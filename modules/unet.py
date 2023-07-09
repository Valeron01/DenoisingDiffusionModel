import math
import typing
import einops
import torch
from torch import nn
from tqdm import trange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels * 4, out_channels, 1)

    def forward(self, x):
        x = nn.functional.pixel_unshuffle(x, 2)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dim=None, groups: int = 8):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * out_channels)
        ) if embed_dim is not None else None

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(groups, out_channels)
        )
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x, embed=None):
        conv1 = self.conv1(x)

        if embed is not None and self.mlp is not None:
            embed = self.mlp(embed)[..., None, None]
            scale, shift = torch.chunk(embed, 2, dim=1)
            conv1 = conv1 * (1 + scale) * shift

        conv1 = self.act1(conv1)

        return self.conv2(conv1)


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.to_qkv = nn.Conv2d(in_channels, n_heads * head_dim * 3, 1, bias=False)
        self.to_output = nn.Conv2d(n_heads * head_dim, out_channels, 1)

    def forward(self, x):
        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = einops.rearrange(q, "b (h c) y x -> b h c (y x)", h=self.n_heads)
        k = einops.rearrange(k, "b (h c) y x -> b h c (y x)", h=self.n_heads)
        v = einops.rearrange(v, "b (h c) y x -> b h c (y x)", h=self.n_heads)

        result = nn.functional.scaled_dot_product_attention(q, k, v)

        result = einops.rearrange(result, "b h c (y x) -> b (h c) y x", h=self.n_heads, x=x.shape[3], y=x.shape[2])

        return self.to_output(result)


class UNet(nn.Module):
    def __init__(
            self,
            num_classes: int,
            n_features_list: typing.List or typing.Tuple = (64, 128, 256),
            use_attention_list: typing.List or typing.Tuple = (False, False, True),
            embedding_dim: int = 256,
    ):
        super().__init__()
        assert len(n_features_list) == len(use_attention_list)

        self.class_embedding = nn.Embedding(num_classes, embedding_dim)

        self.none_class_embedding = nn.Parameter(
            torch.randn(1, embedding_dim), requires_grad=True
        )
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        injection_embedding_dim = 2 * embedding_dim
        self.embedding_proj = nn.Sequential(
            nn.Linear(injection_embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.n_features_list = n_features_list
        self.use_attention_list = use_attention_list

        self.depth = len(n_features_list)

        self.stem = nn.Sequential(
            nn.Conv2d(3, n_features_list[0], 7, 1, 3, bias=False),
            nn.BatchNorm2d(n_features_list[0]),
            nn.LeakyReLU(inplace=True)
        )

        self.encoder = nn.ModuleList()
        for in_features, out_features, use_attention in zip(
                n_features_list[:-1], n_features_list[1:], use_attention_list[:-1]
        ):
            self.encoder.append(nn.ModuleList([
                ResidualBlock(in_features, out_features, injection_embedding_dim),
                ResidualBlock(out_features, out_features, injection_embedding_dim),
                Attention(out_features, out_features) if use_attention else nn.Identity(),
                Downsample(out_features, out_features)
            ]))

        self.middle_block1 = ResidualBlock(
            in_channels=n_features_list[-1], out_channels=n_features_list[-1],
            embed_dim=injection_embedding_dim
        )
        self.middle_block2 = ResidualBlock(
            in_channels=n_features_list[-1], out_channels=n_features_list[-1],
            embed_dim=injection_embedding_dim
        )
        self.middle_attention = Attention(
            in_channels=n_features_list[-1], out_channels=n_features_list[-1]
        ) if use_attention_list[-1] else nn.Identity()

        self.decoder = nn.ModuleList()
        for in_features, out_features, use_attention in zip(
                reversed(n_features_list[1:]), reversed(n_features_list[:-1]), reversed(use_attention_list[:-1])
        ):
            self.decoder.append(nn.ModuleList([
                Upsample(in_features, in_features),
                ResidualBlock(in_features * 2, out_features, injection_embedding_dim),
                ResidualBlock(out_features, out_features, injection_embedding_dim),
                Attention(out_features, out_features) if use_attention else nn.Identity(),
            ]))

        self.final_conv = nn.Conv2d(n_features_list[0] * 2, 3, 1)

    def forward(self, x, t, classes, class_drop_prob=None):
        assert (
                       classes is not None and self.class_embedding is not None
               ) or classes is None, "Classes is not none, but class embedding is"
        stem = self.stem(x)

        time_embed = self.time_embed(t)
        class_embed = self.class_embedding(classes)
        if class_drop_prob:
            class_drop_mask = torch.rand(x.shape[0], device=x.device) < class_drop_prob
            class_drop_mask = class_drop_mask[:, None]

            null_classes = torch.tile(self.none_class_embedding, [x.shape[0], 1])
            class_embed = torch.where(
                class_drop_mask,
                null_classes,
                class_embed
            )

        full_embedding = torch.cat([time_embed, class_embed], dim=1)

        downsample_stage = stem
        downsample_stages = [stem]
        for block1, block2, attention, downsample in self.encoder:
            downsample_stage = block1(downsample_stage, full_embedding)
            downsample_stage = block2(downsample_stage, full_embedding)
            if isinstance(attention, Attention):
                downsample_stage = attention(downsample_stage) + downsample_stage
            downsample_stages.append(downsample_stage)
            downsample_stage = downsample(downsample_stage)

        downsample_stage = self.middle_block1(downsample_stage, full_embedding)
        downsample_stage = self.middle_block2(downsample_stage, full_embedding)
        if isinstance(self.middle_attention, Attention):
            downsample_stage = self.middle_attention(downsample_stage) + downsample_stage

        upsample_stage = downsample_stage
        for previous_stage, (upsample, block1, block2, attention) in zip(reversed(downsample_stages), self.decoder):
            upsample_stage = upsample(upsample_stage)
            upsample_stage = torch.cat([upsample_stage, previous_stage], dim=1)
            upsample_stage = block1(upsample_stage, full_embedding)
            upsample_stage = block2(upsample_stage, full_embedding)
            if isinstance(attention, Attention):
                upsample_stage = attention(upsample_stage)

        return self.final_conv(
            torch.cat([upsample_stage, stem], dim=1)
        )


if __name__ == '__main__':
    unet = UNet(num_classes=10, use_attention_list=(True, True, True)).cuda()

    noise = torch.randn(8, 3, 64, 64).cuda()
    times = torch.randint(0, 1000, (noise.shape[0],)).cuda()
    classes = torch.randint(0, 10, (noise.shape[0],)).cuda()
    res = unet(noise, times, classes, class_drop_prob=0.5)
    print(res.mean())
    del res
