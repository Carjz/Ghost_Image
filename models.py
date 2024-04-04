import torch
import torch.nn as nn
from torchgpipe import GPipe

from constant import *
from functions import *

from pdb import set_trace


# 卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# Transformer层
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_channels)
        self.msa = nn.MultiheadAttention(in_channels, 16)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(),
            nn.Linear(out_channels, in_channels),
        )

    def forward(self, x):
        B, C, H, W = x.size()

        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.layer_norm(x)

        x = x.view(B, H * W, C)
        x = x + self.msa(x, x, x)[0]
        x = x + self.mlp(x)
        x = x.view(B, H, W, C)

        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


# TransUNet架构构建
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 512, stride=2),
            ConvBlock(512, 1024, stride=2),
        )
        if PIPELINE:
            self.encoder = GPipe(
                self.encoder,
                [0, 0, 0, 1, 1, 1, 1, 1],
                chunks=CHUNKS,
                checkpoint="never",
            )

    def forward(self, x):
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        return x, skips


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformers = nn.Sequential(
            *[TransformerBlock(1024, 1024) for _ in range(6)]
        )
        if PIPELINE:
            self.transformers = GPipe(
                self.transformers,
                [0, 0, 1, 1, 1, 1, 1, 1],
                chunks=CHUNKS,
                checkpoint="never",
            )

    def forward(self, x):
        for transformer in self.transformers:
            x = transformer(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            ConvBlock(1024, 512),
            ConvBlock(512, 256),
            ConvBlock(256, 128),
            ConvBlock(128, 64),
        )
        if PIPELINE:
            self.decoder = GPipe(
                self.decoder,
                [0, 0, 0, 0, 1, 1, 1, 1],
                chunks=CHUNKS,
                checkpoint="never",
            )
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x, skips):
        decoder_dim = 1024
        skips = skips[::-1]
        for decode, skip in zip(self.decoder, skips):
            x = upsample(x, skip.size()[-2:])
            x = torch.cat([x, skip], dim=1)

            if x.size(1) != decoder_dim:
                align_layer = ConvBlock(x.size(1), decoder_dim).to(x.device)
                x = align_layer(x)

            x = decode(x)
            x = upsample(x, (IMAGE_SIZE, IMAGE_SIZE))

            decoder_dim //= 2

        return torch.sigmoid(self.output(x))


class TransUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.transformer = Transformer()
        self.decoder = Decoder()

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.transformer(x)
        x = self.decoder(x, skips)
        return x
