import torch
import torchvision.transforms as transforms

from math import ceil, floor


gpus = range(8)
device = torch.device(f"cuda:{gpus[0]}")
device_choice = [
    torch.device(f"cuda:{gpus[1]}"),  # 训练集中采样前的图像
    torch.device(f"cuda:{gpus[1]}"),  # 采样后的图像
    torch.device(f"cuda:{gpus[1]}"),  # 生成随机热光矩阵
    torch.device(f"cuda:{gpus[0]}"),  # 深度图
    torch.device(f"cuda:{gpus[0]}"),  # 深度平面图
]

DATASET_DIR = "Inputs/ShapeNet/ShapeNetCore.v2"
OUTPUT_PATH = "Outputs"
SCANNED_FOLDER = "Inputs/ShapeNet/ShapeNetCore.v2/Cache/Whole_View"
IMAGE_SIZE = 200
BATCH_SIZE = 40
STRIDE = 10
CHUNKS = floor(BATCH_SIZE / len(gpus))

num_epochs = 15
Nyquist_rate = IMAGE_SIZE * IMAGE_SIZE
sampling_times = ceil(Nyquist_rate * 0.01)
test_ratio = 0.1

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
    ]
)

PIPELINE = False
