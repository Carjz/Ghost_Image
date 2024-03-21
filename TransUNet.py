import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import pytorch_msssim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import os
import time
import shutil

IMAGE_SIZE = 32
FOLDER_PATH = "Outputs"
num_epochs = 50

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 双线性插值上采样
def upsample(x, size):
    return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=True)


# 尺寸调整
def adjust_size(x, target_size):
    size = x.size()[-2:]
    if size != target_size:
        x = upsample(x, target_size)
    return x

# 损失函数定义
class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, size_average=True, win_size=11, win_sigma=1.5, channel=3, spatial_dims=2):
        super().__init__()
        self.ssim_loss = pytorch_msssim.SSIM(data_range=data_range, size_average=size_average, win_size=win_size, win_sigma=win_sigma, channel=channel, spatial_dims=spatial_dims)

    def forward(self, img1, img2):
        ssim_value = self.ssim_loss(img1, img2)
        loss = 1 - ssim_value
        return loss

# 卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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
            nn.GELU(),
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
                ConvBlock(1, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256),
                ConvBlock(256, 512),
                ConvBlock(512, 1024),
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
            x = nn.MaxPool2d(2)(x)

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

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 模型实例化
    model = TransUNet().to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = SSIMLoss(channel=1)

    # 训练过程
    for epoch in range(num_epochs):
        train_loss = 0.0
        for images, _ in train_loader:
            I = torch.randn(images.shape)
            I_img = torch.matmul(I, images)
            B = sum(sum(I_img))
            
            ghost_image = B * I

            ghost_image = ghost_image.to(device)
            images = images.to(device)

            outputs = model(ghost_image)

            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.6f}")

    # 保存测试集结果
    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            I = torch.randn(images.shape)
            I_img = torch.matmul(I, images)
            B = sum(sum(I_img))
            
            ghost_image = B * I

            ghost_image = ghost_image.to(device)
            images = images.to(device)

            outputs = model(ghost_image)

            # 保存输出图像
            if os.path.exists(FOLDER_PATH):
                shutil.rmtree(FOLDER_PATH)
            os.makedirs(FOLDER_PATH)
            for i in range(outputs.shape[0]):
                output_image = outputs[i].cpu()
                output_image = torchvision.transforms.ToPILImage()(output_image)
                output_image.save(f"{FOLDER_PATH}/output_{i}.jpg")

    # 保存模型
    torch.save(model.state_dict(), 'model.ckpt')

if __name__ == "__main__":
    start_time = time.time()

    main()

    end_time = time.time()

    during_time = end_time - start_time
    print("All tasks completed.")
    print("Totally takes {:f} seconds.".format(during_time))

