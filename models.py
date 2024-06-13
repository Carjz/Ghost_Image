import torch
import torch.nn as nn
import torch.cuda.amp as amp
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

        x = self.layer_norm(x)

        x = x + self.mlp(x)
        x = x.view(B, H, W, C)

        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


# TransUNet架构构建
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 1024),
        )
        self.conv1 = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(128, 128),
            ConvBlock(256, 256),
            ConvBlock(512, 512),
            ConvBlock(1024, 1024),
        )
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        if PIPELINE:
            pass

    def forward(self, x):
        skips = []
        for conv0, conv1 in zip(self.conv0, self.conv1):
            x = conv0(x)
            x = conv1(x)
            skips.append(x)
            x = self.pooling(x)
        return x, skips


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformers = nn.Sequential(
            *[TransformerBlock(1024, 1024) for _ in range(6)]
        )
        if PIPELINE:
            pass

    def forward(self, x):
        for transformer in self.transformers:
            x = transformer(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(
            ConvBlock(2048, 1024),
            ConvBlock(1024, 512),
            ConvBlock(512, 256),
            ConvBlock(256, 128),
            ConvBlock(128, 64),
        )
        self.conv1 = nn.Sequential(
            ConvBlock(1024, 1024),
            ConvBlock(512, 512),
            ConvBlock(256, 256),
            ConvBlock(128, 128),
            ConvBlock(64, 64),
        )
        self.layer_size = layer_size(self.conv0)
        self.upconv = nn.Sequential(
            nn.Identity(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=3),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=3),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=3),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=3),
        )
        self.croping = nn.Sequential(
            *[
                nn.AdaptiveMaxPool2d(self.layer_size[i].item())
                for i in range(len(self.conv0))
            ]
        )

        delta = IMAGE_SIZE - (self.layer_size[-1] - 4)
        delta0 = delta // 2
        delta -= delta0
        self.pad = nn.ReflectionPad2d((delta0, delta, delta0, delta))
        self.output = nn.Conv2d(64, 1, kernel_size=1)

        if PIPELINE:
            pass

    def forward(self, x, skips):
        skips = skips[::-1]
        for conv0, conv1, croping, upconv, skip, i in zip(
            self.conv0,
            self.conv1,
            self.croping,
            self.upconv,
            skips,
            range(len(self.conv0)),
        ):
            if i != 0:
                x = upconv(x)
            skip = croping(skip)
            x = torch.cat([x, skip], dim=1)
            x = conv0(x)
            x = conv1(x)
        x = self.pad(x)

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
