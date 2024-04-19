import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision.datasets as datasets
import open3d as o3d

import os
import time
import shutil
from math import ceil
from glob import glob
import random

from pdb import set_trace

from constant import *
from models import *
from functions import *


def main():
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

    # 数据加载
    # 查找训练集所需obj文件
    # objs = glob(f"{DATASET_DIR}/**/*.obj", recursive=True)
    # objs = np.load('objs.npy', allow_pickle=True).tolist()

    # 数据集采样，减少训练集规模
    # objs = random.sample(objs, 1000)
    # train_objs = objs
    # test_objs = random.sample(objs, ceil(len(objs)*test_ratio))
    # for obj in test_objs:
    #     train_objs.remove(obj)

    # 创建训练集
    # if not os.path.exists(SCANNED_FOLDER):
    #     os.makedirs(SCANNED_FOLDER)
    # idx = 0
    # for obj in train_objs:
    #     scanned_imgs, _ = scanning(obj)
    #     for img in scanned_imgs:
    #         print_image(img, f"{SCANNED_FOLDER}/{idx}.png")
    #         idx += 1

    train_dataset = datasets.ImageFolder(f"{SCANNED_FOLDER}/..", transform=transform)
    # test_dataset = datasets.ImageFolder(f"{DATASET_DIR}/test", transform=transform)

    # train_dataset = datasets.MNIST(f"Inputs", train=True, download=True, transform=transform)
    # test_dataset = datasets.MNIST(f"Inputs", train=False, download=True, transform=transform)

    # train_dataset = TensorDataset(torch.stack(scanned_imgs))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=16, drop_last=True)
    # test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=16, drop_last=True)
    print('Data loading finished.')

    # 模型实例化
    if PIPELINE:
        model = TransUNet().to(device)
    else:
        model = nn.DataParallel(TransUNet().to(device))

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = SSIMLoss(channel=1)

    model.load_state_dict(torch.load("model.ckpt"))

    # 训练过程
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for images, _ in train_loader:
            # 加载图像
            images = images.to(device)

            sampled_images = sampling(normalize(images))

            # 前向传播
            outputs = model(sampled_images)

            loss = criterion(outputs, images)
            optimizer.zero_grad()

            # 反向传播
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.6f}", flush=True)

        torch.save(model.state_dict(), "model.ckpt")


if __name__ == "__main__":
    start_time = time.time()

    main()

    end_time = time.time()

    during_time = end_time - start_time
    print("All tasks completed.")
    print("Totally takes {:f} seconds.\n".format(during_time))
