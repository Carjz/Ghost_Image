import torch
from math import ceil, floor
import torchvision.transforms as transforms


host_id = 0
device = torch.device(f"cuda:{host_id}")
gpus = [0, 1, 2]

DATASET_DIR = "Inputs/Caltech_256"
FOLDER_PATH = "Outputs"
IMAGE_SIZE = 200
BATCH_SIZE = 64
CHUNKS = floor(BATCH_SIZE / len(gpus))

num_epochs = 3
Nyquist_rate = IMAGE_SIZE * IMAGE_SIZE
sampling_times = ceil(Nyquist_rate * 0.01)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
    ]
)

PIPELINE = False
