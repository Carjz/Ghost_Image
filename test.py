import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


import os
import time
import shutil
import random

from pdb import set_trace

from constant import *
from models import *
from functions import *


def main_eval():
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

    objs = np.load("objs.npy", allow_pickle=True).tolist()
    print(len(objs))

    # 数据集采样，减少训练集规模
    # objs = random.sample(objs, 1000)
    test_objs = random.sample(objs, ceil(len(objs) * test_ratio))

    # 模型实例化
    # if PIPELINE:
    #     model = TransUNet().to(device)
    # else:
    #     model = nn.DataParallel(TransUNet().to(device))
    # model.load_state_dict(torch.load("Models/model-iter_0.ckpt"))

    # 定义优化器和损失函数
    criterion = SSIMLoss(channel=1)

    # 保存测试集结果
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)

    # model.eval()
    idx = 0
    eval_loss = 0.0
    eval_time = 0.0

    with torch.no_grad():
        for obj in test_objs:
            set_trace()
            scanned_img, _ = scanning(obj)
            scanned_img = scanned_img.to(device_choice[0]).unsqueeze(0)

            start_time = time.time()

            sampled_image = normalize(sampling(normalize(scanned_img)))

            # # 前向传播
            # outputs = model(sampled_image)
            # outputs_vis = normalize(outputs)

            end_time = time.time()
            eval_time += end_time - start_time
            loss = criterion(sampled_image, scanned_img)

            eval_loss += loss.item()
            print(loss.item())

            # 保存输出图像
            print_image(sampled_image[0], f"{OUTPUT_PATH}/{idx}.png")
            idx += 1

    print(
        "Model evaluation takes {:f} seconds per image.".format(
            eval_time / (BATCH_SIZE * len(test_objs))
        )
    )
    print(f"Model evaluation loss: {eval_loss/len(test_objs):.6f}\n")


if __name__ == "__main__":
    start_time = time.time()

    main_eval()

    end_time = time.time()

    during_time = end_time - start_time
    print("All tasks completed.")
    print("Totally takes {:f} seconds.\n".format(during_time))
