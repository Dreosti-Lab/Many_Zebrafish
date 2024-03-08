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

    def classify(self, video, roi, stimulus_frame):
        data = generate_data(video, roi, self.input_dim, self.input_times, stimulus_frame)
        input = torch.tensor((2.0 * (np.float32(data) / 255.0)) - 1.0)
        input = torch.unsqueeze(input, 0)
        input = input.to(self.device)
        output = self.model(input)
        output = (output.cpu().detach().numpy()[0][0] > 0.5)
        return output

# Utilities for working with 96-well plate experiments

# Generate classifier data
def generate_data(video, roi, dim, times, stimulus_frame):
    data = np.zeros((3, dim, dim), dtype=np.uint8)
    for f in range(3):
        ret = video.set(cv2.CAP_PROP_POS_FRAMES, stimulus_frame+times[f])
        ret, frame = video.read()
        crop = MZV.get_ROI_crop(frame, roi)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (dim,dim))
        data[f,:,:] = resize
    return data

#FIN
