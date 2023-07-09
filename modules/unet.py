import typing
import einops
import torch
from torch import nn


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

        if embed and self.mlp:
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
            n_features_list: typing.List or typing.Tuple = (64, 128, 256),
            use_attention_list: typing.List or typing.Tuple = (False, False, True),
            embedding_dim: int = 256,
    ):
        super().__init__()
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
                ResidualBlock(in_features, out_features, embedding_dim * 2),
                ResidualBlock(out_features, out_features, embedding_dim * 2),
                Attention(out_features, out_features) if use_attention else nn.Identity(),
                Downsample(out_features, out_features)
            ]))

        self.middle_block1 = ResidualBlock(
            in_channels=n_features_list[-1], out_channels=n_features_list[-1],
            embed_dim=embedding_dim * 2
        )
        self.middle_block2 = ResidualBlock(
            in_channels=n_features_list[-1], out_channels=n_features_list[-1],
            embed_dim=embedding_dim * 2
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
                ResidualBlock(in_features * 2, out_features, embedding_dim * 2),
                ResidualBlock(out_features, out_features, embedding_dim * 2),
                Attention(out_features, out_features) if use_attention else nn.Identity(),
            ]))

        self.final_conv = nn.Conv2d(n_features_list[0] * 2, 3, 1)

    def forward(self, x):
        stem = self.stem(x)

        downsample_stage = stem
        downsample_stages = [stem]
        for block1, block2, attention, downsample in self.encoder:
            downsample_stage = block1(downsample_stage)
            downsample_stage = block2(downsample_stage)
            downsample_stage = attention(downsample_stage) + downsample_stage
            downsample_stages.append(downsample_stage)
            downsample_stage = downsample(downsample_stage)

        downsample_stage = self.middle_block1(downsample_stage)
        downsample_stage = self.middle_block2(downsample_stage)
        downsample_stage = self.middle_attention(downsample_stage) + downsample_stage

        upsample_stage = downsample_stage
        for previous_stage, (upsample, block1, block2, attention) in zip(reversed(downsample_stages), self.decoder):
            upsample_stage = upsample(upsample_stage)
            upsample_stage = torch.cat([upsample_stage, previous_stage], dim=1)
            upsample_stage = block1(upsample_stage)
            upsample_stage = block2(upsample_stage)
            upsample_stage = attention(upsample_stage)

        return self.final_conv(
            torch.cat([upsample_stage, stem], dim=1)
        )


if __name__ == '__main__':
    noise = torch.randn(1, 3, 256, 256)

    unet = UNet()
    res = unet(noise)
    print(res.shape)
