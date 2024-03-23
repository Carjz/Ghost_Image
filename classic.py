import numpy as np
import cv2 as cv
import torch
import torchvision

import os
import shutil

from skimage.metrics import structural_similarity as ssim
import time

import multiprocessing as mulp

input_folder = "Inputs/local_MNIST"
output_folder = "Outputs/classic"
iteration = 65536 * 16  # 2^16 设定帧数，最大为哈达玛矩阵的 行/列 数
nthreads = 65  # 设置进程数
px = 64  # 像素


# 成像函数
def imaging(input_file):
    start_time = time.time()

    input_dir_path = os.path.dirname(input_file)
    filename = os.path.basename(input_file)

    img = cv.imread(input_file, 0)  # 读取图片,0代表灰度
    img = cv.resize(img, dsize=(px, px))
    cv.threshold(img, 128, 255, cv.THRESH_BINARY)  # 转化为二值图

    Sum_of_B = 0  # 定义桶测量值的和
    Sum_of_I = np.zeros(
        img.shape
    )  # 定义热光矩阵的和，初始矩阵设为零，且尺寸为 img.shape 的尺寸
    Sum_of_BI = np.zeros(
        img.shape
    )  # 定义B*I矩阵的和，初始矩阵设为零，且尺寸为 img.shape 的尺寸（B是桶测量值，I是热光矩阵（照明图案））

    for it in range(iteration):
        # 进度提示语句
        # if it % 10000 == 0:
        #    print(
        #        "Thread {:s} - {:.2%}".format(
        #            mulp.current_process().name[-1], it / iteration
        #        )
        #    )

        I = np.random.normal(0, 1, img.shape)  # 生成热光矩阵（照明图案）
        Sum_of_I = Sum_of_I + I  # 照明图案累加
        I_img = np.multiply(
            I, img
        )  # 照明图案乘以图像矩阵，即模拟热光照射物体的过程，I_img即为照射物体后的光束矩阵

        B = sum(sum(I_img))  # 对照射物体后的光束矩阵求和，即获得桶测量值

        Sum_of_B = Sum_of_B + B  # 桶测量值累加
        Sum_of_BI = Sum_of_BI + B * I  # B*I矩阵累加（即利用照明图案和桶测量值得相关性）

    Avg_B = Sum_of_B / iteration  # 桶测量值的均值
    Avg_I = Sum_of_I / iteration  # 热光矩阵的均值
    Avg_BI = Sum_of_BI / iteration  # B*I矩阵的均值

    GI = Avg_BI - Avg_I * Avg_B  # 计算鬼成像

    # 重建图像的像素值归一化
    mi = np.min(GI)
    mx = np.max(GI)
    GI = 255 * (GI - mi) / (mx - mi)

    end_time = time.time()
    during_time = end_time - start_time

    # 评估成像质量
    score = ssim(img, GI, data_range=GI.max() - GI.min())

    # 导出生成的图像
    output_path = os.path.join(output_folder, os.path.basename(input_dir_path))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv.imwrite(os.path.join(output_path, filename + ".bmp"), GI)

    # 每个进程完成关联计算后，输出自己的计算时间和计算质量
    print(
        "Thread {:s} Completed: Time={:f} second, Score={:f}.\n".format(
            mulp.current_process().name[15:],
            during_time,
            score,
        )
    )


# 遍历函数
def traverse_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            yield file_path


def main():
    start_time = time.time()

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    input_files = traverse_files(input_folder)

    # 并行池分配
    print("Creating parallel pool...")
    pool = mulp.Pool(processes=nthreads)
    print("Done.\n")
    print("Start parallel computing.\n")
    pool.map(imaging, input_files)
    # for _ in pool.imap(imaging, input_files):
    #     pass
    pool.close()
    pool.join()

    # 程序总用时
    end_time = time.time()
    during_time = end_time - start_time
    print("All tasks completed.")
    print("Totally takes {:f} second.".format(during_time))


if __name__ == "__main__":
    main()

