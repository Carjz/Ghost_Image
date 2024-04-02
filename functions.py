import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pytorch_msssim

from math import sqrt, inf

from constant import *

from pdb import set_trace


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

    return x


# 图像采样
def sampling(images):
    I = torch.randn(
        BATCH_SIZE, sampling_times, IMAGE_SIZE, IMAGE_SIZE, device=f"cuda:{gpus[1]}"
    ).to(
        device
    )  # 热光矩阵/随机散斑图案speckle
    I_imgs = I * images  # 散斑与物体作用
    B = I_imgs.sum(dim=(2, 3), keepdim=True)  # 桶测量值
    BI = B * I  # 桶测量值与散斑相关性

    B_avg = B.sum(dim=1, keepdim=True) / sampling_times
    I_avg = I.sum(dim=1, keepdim=True) / sampling_times
    BI_avg = BI.sum(dim=1, keepdim=True) / sampling_times

    # FISTA_image = FISTA(BI_avg, B_avg * I_avg, _lambda=0.05, _alpha=0.1, K=5)
    sampled_images = BI_avg - B_avg * I_avg

    return sampled_images


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


def print_image(image, filename):
    image = image.cpu()
    image = transforms.ToPILImage()(image)
    image.save(filename)


# 软阈值函数
def soft_threshold(x, _lambda):
    return torch.sign(x) * torch.clamp(torch.abs(x) - _lambda, min=0)


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
