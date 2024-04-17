import torch
from math import ceil, floor
import torchvision.transforms as transforms


gpus = range(8)
device = torch.device(f"cuda:{gpus[0]}")

DATASET_DIR = "Inputs/ShapeNet/ShapeNetCore.v2"
FOLDER_PATH = "Outputs"
SCANNED_FOLDER = "Inputs/ShapeNet/ShapeNetCore.v2/Cache/None_Type"
IMAGE_SIZE = 200
BATCH_SIZE = 40
STRIDE = 10
CHUNKS = floor(BATCH_SIZE / len(gpus))

num_epochs = 10
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
