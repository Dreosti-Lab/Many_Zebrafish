# -*- coding: utf-8 -*-
"""
Many_Zebrafish: Fish Class

@author: kamnpff (Adam Kampff)
"""
# Import useful libraries
import numpy as np

# Fish Class
class Fish:
    def __init__(self, _id):
        self.id = _id
        self.ul = None
        self.lr = None
        self.width = None
        self.height = None
        self.x = []
        self.y = []
        self.area = []
        self.motion = []
        self.threshold = None
        self.background = None

    def set_roi(self, _ul, _lr):
        self.ul = (int(round(_ul[0])), int(round(_ul[1])))   # ROI upper right corner
        self.lr = (int(round(_lr[0])), int(round(_lr[1])))   # ROI lower right corner
        self.width = self.lr[0] - self.ul[0]
        self.height = self.lr[1] - self.ul[1]
        self.background = np.zeros((self.width, self.height))
        return

    def set_background(self, image):
        r1 = self.ul[1]
        r2 = self.lr[1]
        c1 = self.ul[0]
        c2 = self.lr[0]
        self.background = image[r1:r2, c1:c2]
        self.threshold = np.median(self.background[:])/7

# Utilities for working with Fish Class

# Create Plate
def create_plate():
    plate = []
    for i in range(96):
        fish = Fish(i+1)
        plate.append(fish)
    return plate        

# Set backgrounds
def set_backgrounds(background, plate):
    for fish in plate:
        fish.set_background(background)
    return plate

# FIN
