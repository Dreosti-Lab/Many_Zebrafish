# -*- coding: utf-8 -*-
"""
Many_Zebrafish: Classifier Class

@author: kampff
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision import transforms
from torchsummary import summary

# Import local modules
import MZ_video as MZV

# Locals libs
import classifier.model as model

# Classifier Class
class Classifier:
    def __init__(self, _model_path):
        self.model_path = _model_path
        self.model = model.custom()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model = self.model.eval()
        self.input_dim = 224
        self.input_times = [0, 9, 19]
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.model.to(self.device)
        return

    def classify(self, datapoint, debug=False):
        input = torch.tensor((2.0 * (np.float32(datapoint) / 255.0)) - 1.0)
        input = torch.unsqueeze(input, 0)
        input = input.to(self.device)
        output = self.model(input)
        output = (output.cpu().detach().numpy()[0][0] > 0.5)
        if debug:
            if output:
                plt.title("Response")
            else:
                plt.title("No Response")
            feature = np.swapaxes(datapoint, 2, 0)
            plt.imshow(feature)
            plt.show()
        return output

# Utilities for working with 96-well plate experiments

# Generate classifier datapoint
def generate_datapoint(frames, roi, data_dim, data_times, index_frame):
    data = np.zeros((3, data_dim, data_dim), dtype=np.uint8)
    for f in range(3):
        frame = frames[index_frame + data_times[f]]
        crop = MZV.get_ROI_crop(frame, roi)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (data_dim, data_dim))
        data[f,:,:] = resize
    return data

#FIN
