import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch import inf
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pytorch_msssim

from math import ceil, sqrt
import os
import time
import shutil

from pdb import set_trace


FOLDER_PATH = "Outputs"
IMAGE_SIZE = 28
BATCH_SIZE = 64

num_epochs = 50
Nyquist_rate = IMAGE_SIZE * IMAGE_SIZE
sampling_times = ceil(Nyquist_rate * 0.01)

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


# 双线性插值上采样
def upsample(x, size):
    return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=True)


# 尺寸调整
def adjust_size(x, target_size):
    size = x.size()[-2:]
    if size != target_size:
        x = upsample(x, target_size)
    return x


# 归一化
def normalize(x):
    mx = x.max(3, True).values.max(2, True).values
    mn = x.min(3, True).values.min(2, True).values
    diff = mx - mn
    diff[diff == 0] = 1
    x = (x - mn) / diff
    x[x == inf] = 0

    return x * 255


# 损失函数定义
class SSIMLoss(nn.Module):
    def __init__(
        self,
        data_range=1.0,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
    ):
        super().__init__()
        self.ssim_loss = pytorch_msssim.SSIM(
            data_range=data_range,
            size_average=size_average,
            win_size=win_size,
            win_sigma=win_sigma,
            channel=channel,
            spatial_dims=spatial_dims,
        )

    def forward(self, img1, img2):
        ssim_value = self.ssim_loss(img1, img2)
        loss = 1 - ssim_value
        return loss


# 卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# 软阈值函数
def soft_threshold(x, _lambda):
    thresholded = torch.abs(x) - _lambda
    thresholded = torch.clamp(thresholded, min=0)
    return torch.sign(x) * thresholded


# FISTA图像重建
def FISTA(y, H, _lambda=0.01, _alpha=1, K=10):
    H_T = H.transpose(2, 3)
    x = torch.zeros_like(y).float()
    x_old = []
    t = 1

    for k in range(K):
        x_old = x.clone()
        grad = H_T @ (H @ x - y)
        x = soft_threshold(x - _alpha * grad, _lambda * _lambda)
        t_new = (1 + sqrt(1 + 4 * t * t)) / 2
        x = x + (t - 1) / t_new * (x - x_old)
        t = t_new

    x = x.nan_to_num(0)

    return x


# 图像采样
def sampling(images):
    I = torch.tensor([])
    for iter in range(sampling_times):
        I_tmp = torch.randn(images.shape)  # 热光矩阵/随机散斑图案speckle
        I = torch.cat([I, I_tmp], dim=1)

    I = I.to(device)
    I_imgs = I * images  # 散斑与物体作用
    B = I_imgs.sum(dim=(2, 3), keepdim=True)  # 桶测量值
    BI = B * I  # 桶测量值与散斑相关性

    B_avg = B.sum(dim=1, keepdim=True) / sampling_times
    I_avg = I.sum(dim=1, keepdim=True) / sampling_times
    BI_avg = BI.sum(dim=1, keepdim=True) / sampling_times

    FISTA_image = FISTA(BI_avg, B_avg * I_avg, _lambda=0.05, _alpha=0.1, K=5)

    return FISTA_image


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


class TransUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                ConvBlock(1, 64, stride=2),
                ConvBlock(64, 128, stride=2),
                ConvBlock(128, 256, stride=2),
                ConvBlock(256, 512, stride=2),
                ConvBlock(512, 1024, stride=2),
            ]
        )
        self.transformers = nn.ModuleList(
            [TransformerBlock(1024, 1024) for _ in range(6)]
        )
        self.decoder = nn.ModuleList(
            [
                ConvBlock(1024, 512),
                ConvBlock(512, 256),
                ConvBlock(256, 128),
                ConvBlock(128, 64),
            ]
        )
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
            # if min([x.size(-1), x.size(-2)]) >= 8:
            #     x = nn.MaxPool2d(2)(x)

        for transformer in self.transformers:
            x = transformer(x)

        decoder_dim = 1024
        skips = skips[::-1]
        for decode, skip in zip(self.decoder, skips):
            x = upsample(x, skip.size()[-2:])
            x = torch.cat([x, skip], dim=1)

            if x.size(1) != decoder_dim:
                align_layer = ConvBlock(x.size(1), decoder_dim).to(device)
                x = align_layer(x)

            x = decode(x)
            x = upsample(x, (IMAGE_SIZE, IMAGE_SIZE))

            decoder_dim //= 2

        return torch.sigmoid(self.output(x))


def print_image(image, filename):
    image = image.cpu()
    image = transforms.ToPILImage()(image)
    image.save(filename)


def main():
    # 数据加载
    train_dataset = datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            ]
        ),
    )
    test_dataset = datasets.MNIST(
        "data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            ]
        ),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 模型实例化
    model = TransUNet().to(device)

    # 定义优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    criterion = SSIMLoss(channel=1)

    # 训练过程
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)

            FISTA_images = sampling(images)

            FISTA_images = normalize(FISTA_images)

            # 前向传播
            outputs = model(FISTA_images)

            loss = criterion(outputs, images)
            optimizer.zero_grad()

            # 反向传播
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.6f}")

    # 保存测试集结果
    model.eval()
    if os.path.exists(FOLDER_PATH):
        shutil.rmtree(FOLDER_PATH)
    os.makedirs(FOLDER_PATH)
    idx = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            FISTA_images = sampling(images)

            FISTA_images = normalize(FISTA_images)

            # 前向传播
            outputs = model(FISTA_images)

            # 保存输出图像
            for i in range(outputs.shape[0]):
                print_image(outputs[i], f"{FOLDER_PATH}/{labels[i].item()}_{idx}.jpg")
                idx += 1

    # 保存模型
    torch.save(model.state_dict(), "model.ckpt")


if __name__ == "__main__":
    start_time = time.time()

    main()

    end_time = time.time()

    during_time = end_time - start_time
    print("All tasks completed.")
    print("Totally takes {:f} seconds.".format(during_time))

