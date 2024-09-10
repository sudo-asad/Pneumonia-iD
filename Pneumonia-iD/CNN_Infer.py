#Import necessary modules
import torch
from PIL import Image as pil_image
from PySide6 import QtCore
import traceback
from os import path, getcwd
from torchvision import transforms
from torch.nn import Conv2d, ReLU, MaxPool2d, BatchNorm2d, Dropout2d, LazyLinear
from Utils import get_model_dir

MODEL_PATH = path.join(get_model_dir(), "model_state.pth")

IMAGE_SIZE = (256, 256)


class Predict(QtCore.QThread):

    #Signals to update the main thread about status
    signal_status = QtCore.Signal(list)

    #Done status
    signal_done = QtCore.Signal(str)

    def __init__(self, file_list):
        super(Predict, self).__init__()
        self.file_list = file_list
        print("Classifying files: {}".format(file_list))


    def run(self):

        try:
            transformer_function_infer = transforms.Compose(
                [transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Resize(IMAGE_SIZE[0])])

            device = "cpu"

            infer_model = CNN_Custom(1)
            infer_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            infer_model.eval()

            with torch.no_grad():
                for current_file in self.file_list:
                    current_image = pil_image.open(path.normpath(current_file))
                    transformed = transformer_function_infer(current_image)

                    transformed = transformed.unsqueeze(0)
                    transformed = transformed.to(device)

                    y_out = infer_model.forward(transformed)
                    y_out = y_out.squeeze(1)
                    prediction = torch.special.expit(y_out).detach().numpy().tolist()[0]
                    self.signal_status.emit([current_file, prediction])
                    print("For file {} prediction: {} ".format(current_file, prediction))


            self.signal_done.emit("done")

        except Exception as e:
            with open(path.join(getcwd(), "Error_log.txt"), "at") as wf:
                wf.write(traceback.format_exc())

            self.signal_done.emit("error")

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



