import torch
from math import ceil
import torchvision.transforms as transforms


DATASET_DIR = "Inputs/Caltech_256"
FOLDER_PATH = "Outputs"
IMAGE_SIZE = 200
BATCH_SIZE = 64

num_epochs = 3
Nyquist_rate = IMAGE_SIZE * IMAGE_SIZE
sampling_times = ceil(Nyquist_rate * 0.01)

device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
gpus = [0, 1, 2]

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
    ]
)
