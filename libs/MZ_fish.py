# -*- coding: utf-8 -*-
"""
Many_Zebrafish: Fish Class

@author: kampff
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
        self.heading = []
        self.area = []
        self.motion = []
        self.threshold_background = None
        self.threshold_motion = None
        self.background = None
        self.background_stack = None
        self.stack_size = 20
        self.stack_count = 0
        self.frames_since_background_update = 0
        self.previous = None
        self.previous_motion = 0
        return

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
        self.threshold_background = np.median(self.background[:])/15
        self.threshold_motion = np.median(self.background[:])/20
        self.background_stack = np.zeros((self.height, self.width, self.stack_size), dtype = np.uint8)
        for i in range(self.stack_size):
            self.background_stack[:,:,i] = self.background
        self.stack_count = 0
        return

    def update_background(self, crop):
        self.background_stack[:,:,self.stack_count] = crop
        self.stack_count = (self.stack_count + 1) % self.stack_size
        self.frames_since_background_update = 0
        tmp = np.uint8(np.median(self.background_stack, axis = 2))
        self.background = np.copy(tmp)
        return

    def add_behaviour(self, x, y, heading, area, motion):
        self.x.append(x)
        self.y.append(y)
        self.heading.append(heading)
        self.area.append(area)
        self.motion.append(motion)
        return

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

# Save plate (fish behaviour and intensity roi)
def save_plate(plate, intensity, output_folder):
    num_frames = len(intensity)

    # Save fish behaviour
    for i, fish in enumerate(plate):
        fish_path = output_folder + f'/fish/{(i+1):02d}_fish.csv'
        fish_behaviour = np.zeros((num_frames,5), dtype=np.float32)
        fish_behaviour[:,0] = np.array(fish.x)
        fish_behaviour[:,1] = np.array(fish.y)
        fish_behaviour[:,2] = np.array(fish.area)
        fish_behaviour[:,3] = np.array(fish.heading)
        fish_behaviour[:,4] = np.array(fish.motion)
        np.savetxt(fish_path, fish_behaviour, fmt='%.3f', delimiter=',')

    # Save intensity
    intensity_path = output_folder + '/intensity.csv'
    intensity_array = np.array(intensity)
    np.savetxt(intensity_path, intensity_array, fmt='%d')

    return

# FIN
