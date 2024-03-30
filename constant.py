import torch
from math import ceil


FOLDER_PATH = "Outputs"
IMAGE_SIZE = 128
BATCH_SIZE = 100

num_epochs = 50
Nyquist_rate = IMAGE_SIZE * IMAGE_SIZE
sampling_times = ceil(Nyquist_rate * 0.01)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            ])
