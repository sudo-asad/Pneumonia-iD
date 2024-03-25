#Import necessary modules
import torch
import time
from PIL import Image as pil_image

from os import path, walk
from torchvision import datasets, transforms
from torch.nn import Conv2d, ReLU, MaxPool2d, Sequential, BatchNorm2d, Dropout2d, Sigmoid, LazyLinear, BCEWithLogitsLoss
import random

DATASET_PATH = "shuffled_dataset"
IMAGE_SIZE = (256, 256)

class CNN_Custom(torch.nn.Module):

    def __init__(self, input_channels, padding=1):
        super(CNN_Custom, self).__init__()
        self.layer_cnn = torch.nn.Sequential(
            Conv2d(input_channels, 16, kernel_size=3, padding=padding, groups=input_channels),
            ReLU(),
            Conv2d(16, 16, kernel_size=3, padding=padding, groups=input_channels),
            ReLU(),
            Conv2d(16, 32, kernel_size=3, padding=padding, groups=input_channels),
            ReLU(),
            Conv2d(32, 64, kernel_size=3, padding=padding, groups=input_channels),
            ReLU(),
            Conv2d(64, 256, kernel_size=3, padding=padding, groups=input_channels),
            ReLU(),
            BatchNorm2d(256),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(256, 256, kernel_size=3, padding=padding, groups=input_channels),
            ReLU(),
            Dropout2d(0.3))

        #For 2 classes
        self.fully_connected = LazyLinear(1)

    def forward(self, x):
        x = self.layer_cnn(x)
        x = x.view(x.size(0), -1)

        x = self.fully_connected(x)
        return x


# ---- Inference ---- #

# Any inference images must be converted to the same size and images used in training: 256x256
INFER_DIR = path.join(DATASET_PATH, "testing")

start_time = time.time()
transformer_function_infer = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Resize(IMAGE_SIZE[0])])

# Get all files from the root and sub directories:
files_to_infer = []
for root, directories, file_list in walk(INFER_DIR):
    files_to_infer.extend([path.join(root, x) for x in file_list])

# Shuffle the images, because why not, and use a short sample size from it for inference
random.shuffle(files_to_infer)
INFER_MAX_COUNT = 20

device = "cpu"
# output_channels for init obsolete after experimentation.
infer_model = CNN_Custom(1, 1)
infer_model.load_state_dict(torch.load("Model/model_state.pth", map_location="cpu"))
infer_model.eval()

with torch.no_grad():
    for current_file in files_to_infer[:INFER_MAX_COUNT]:
        current_image = pil_image.open(current_file)
        transformed = transformer_function_infer(current_image)
        
        transformed = transformed.unsqueeze(0)
        transformed = transformed.to(device)
        
        y_out = infer_model.forward(transformed)
        y_out = y_out.squeeze(1)
        prediction = torch.special.expit(y_out).detach().numpy().tolist()[0]
        if prediction > 0.5:
            pred_string = "Pneumonia"
        else:
            pred_string = "Healthy"
        
        print("For file: {} -- Prediction is: {} -- Raw score is: {}".format(current_file, pred_string, prediction))

print("Time taken to infer {} images: {}".format(INFER_MAX_COUNT, time.time() - start_time))
