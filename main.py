import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


import os
import time
import shutil

from pdb import set_trace

from constant import *
from models import *
from functions import *


def main():
    # 数据加载
    train_dataset = datasets.MNIST(f"Inputs", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(f"Inputs", train=False, download=True, transform=transform)
    # train_dataset = datasets.ImageFolder(f"{DATASET_DIR}/train", transform=transform)
    # test_dataset = datasets.ImageFolder(f"{DATASET_DIR}/test", transform=transform)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)

    # 模型实例化
    if PIPELINE:
        model = TransUNet().to(device)
    else:
        model = nn.DataParallel(TransUNet().to(device))

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = SSIMLoss(channel=1)

    # 训练过程
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)

            sampled_images = sampling(images) / 255

            # 前向传播
            outputs = model(sampled_images)

            loss = criterion(outputs, images)
            optimizer.zero_grad()

            # 反向传播
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.6f}")

    # 保存测试集结果
    if os.path.exists(FOLDER_PATH):
        shutil.rmtree(FOLDER_PATH)
    os.makedirs(FOLDER_PATH)

    model.eval()
    idx = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            sampled_images = sampling(images) / 255

            # 前向传播
            outputs = model(sampled_images)
            outputs = normalize(outputs) * 255

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
