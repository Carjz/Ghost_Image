import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pytorch_msssim
import open3d as o3d
import numpy as np

from math import inf, ceil
import os
import sys

from constant import *

from pdb import set_trace


# 归一化
def normalize(x):
    mx = x.max(-1, True).values.max(-2, True).values
    mn = x.min(-1, True).values.min(-2, True).values
    diff = mx - mn
    diff[diff == 0] = 1.0
    x = (x - mn) / diff
    x[x == inf] = 0.0

    return x


# 图像采样
def block_sampling(images, idx, dev, B_tot, I_tot, BI_tot):
    dev_idx = gpus.index(dev)
    if idx > sampling_times:
        cur = sampling_times % SAMPLING_ITERATION
    else:
        cur = SAMPLING_ITERATION

    I = torch.randn(
        images.size(0),
        cur,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=f"cuda:{dev}",
    )  # 热光矩阵/随机散斑图案speckle
    I_imgs = I * images.to(I.device)  # 散斑与物体作用
    B = I_imgs.sum(dim=(2, 3), keepdim=True)  # 桶测量值
    BI = B * I  # 桶测量值与散斑相关性

    B_tot[dev_idx] += B.sum(dim=1, keepdim=True).to(device_choice[2])
    I_tot[dev_idx] += I.sum(dim=1, keepdim=True).to(device_choice[2])
    BI_tot[dev_idx] += BI.sum(dim=1, keepdim=True).to(device_choice[2])


def sampling(images):
    iter_times = ceil(sampling_times / SAMPLING_ITERATION)
    idx = 0
    B_tot = torch.zeros(len(gpus), BATCH_SIZE, 1, 1, 1).to(device_choice[2])
    I_tot = torch.zeros(len(gpus), BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE).to(
        device_choice[2]
    )
    BI_tot = torch.zeros(len(gpus), BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE).to(
        device_choice[2]
    )

    for _ in range(ceil(iter_times / len(gpus))):
        for dev in gpus:
            idx += SAMPLING_ITERATION
            block_sampling(images, idx, dev, B_tot, I_tot, BI_tot)

    B_tot = B_tot.sum(dim=0, keepdim=False) / sampling_times
    I_tot = I_tot.sum(dim=0, keepdim=False) / sampling_times
    BI_tot = BI_tot.sum(dim=0, keepdim=False) / sampling_times

    sampled_images = BI_tot - B_tot * I_tot

    return sampled_images.to(device_choice[1])


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
        self.ssim_loss = pytorch_msssim.MS_SSIM(
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


def scanning(obj):
    mesh = o3d.io.read_triangle_mesh(obj, print_progress=False)
    mesh = mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()

    # 计算物体的包围盒
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_extent = bbox.get_extent()

    camera_distance = max(bbox_extent) * (4.0 / 5.0)
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic.set_intrinsics(
        IMAGE_SIZE,
        IMAGE_SIZE,
        IMAGE_SIZE / 2,
        IMAGE_SIZE / 2,
        IMAGE_SIZE / 2,
        IMAGE_SIZE / 2,
    )
    cam.extrinsic = np.array(
        [[0, 0, -1, 0], [0, -1, 0, 0], [-1, 0, 0, camera_distance], [0, 0, 0, 1]]
    )

    # 离屏渲染
    vis = o3d.visualization.rendering.OffscreenRenderer(IMAGE_SIZE, IMAGE_SIZE)
    vis.setup_camera(cam.intrinsic, cam.extrinsic)
    vis.scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS,
        np.array([0, 0, 0]),
    )
    vis.scene.add_geometry("mesh", mesh, o3d.visualization.rendering.MaterialRecord())

    depth = vis.render_to_depth_image()
    depth = torch.from_numpy(np.asarray(depth)).to(device_choice[3]).unsqueeze(0)

    depth = normalize(1 - depth)
    binary_depth = depth.clone()
    binary_depth[binary_depth != 0] = 1.0

    # depth = normalize(1 - depth) * IMAGE_SIZE * STRIDE
    # depth = (depth / STRIDE).ceil()
    # depth = normalize(depth)

    # unique_depths = depth.unique()
    # unique_depths = unique_depths[unique_depths != 0]
    # depth_planes = []

    # for d in unique_depths:
    #     plane = depth * (depth == d)
    #     plane[plane != 0] = 1.0

    #     depth_planes.append(plane.to(device_choice[4]))

    return (binary_depth, depth)


def layer_size(layers):
    sz = IMAGE_SIZE
    l = len(layers)
    szs = torch.zeros(l, dtype=int)

    for i in range(l):
        sz = (sz - 4) // 2
    for i in range(l):
        szs[i] = sz
        sz = (sz - 4) * 3
    return szs


def print_image(image, filename):
    image = image.cpu()
    image = transforms.ToPILImage()(image)
    image.save(filename)


def block_print():
    sys.stdout = open(os.devnull, "w")
    # sys.stderr = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__
    # sys.stderr = sys.__stderr__
