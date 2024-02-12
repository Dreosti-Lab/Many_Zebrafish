# -*- coding: utf-8 -*-
"""
Many_Zebrafish: Fish Class

@author: kampff
"""
# Import libraries
import numpy as np

# Fish Class
class Fish:
    def __init__(self):
        self.roi_ul = None                  # ROI: upper left coordinate (x,y)
        self.roi_lr = None                  # ROI: lower right coordinate (x,y)
        self.roi_width = None               # ROI: width (pixels)
        self.roi_height = None              # ROI: height (pixels)
        self.x = []                         # List of fish centroid X positions
        self.y = []                         # List of fish centroid Y positions
        self.heading = []                   # List of fish heading angles (0 right, 90 up)
        self.area = []                      # List of fish particle areas
        self.motion = []                    # List of fish "motion" values
        self.threshold_background = None    # Threshload level for background segmentation
        self.threshold_motion = None        # Threshold level for motion detection
        self.background = None              # Background frame (roi_width xx roi_height)
        self.stack = None                   # Frame history stack
        self.stack_size = 20                # Sice of history stack
        self.stack_count = 0                # Current position to update in history stack
        self.since_stack_update = 0         # Frames since last history stack update
        self.previous = None                # Previous frame (roi_width xx roi_height)
        self.previous_motion = 0            # Previous "motion" value
        return

    def set_roi(self, _ul, _lr):
        self.ul = (int(round(_ul[0])), int(round(_ul[1])))   # ROI upper right corner
        self.lr = (int(round(_lr[0])), int(round(_lr[1])))   # ROI lower right corner
        self.width = self.lr[0] - self.ul[0]
        self.height = self.lr[1] - self.ul[1]
        self.background = np.zeros((self.width, self.height))
        return

    def init_background(self, image):
        r1 = self.ul[1]
        r2 = self.lr[1]
        c1 = self.ul[0]
        c2 = self.lr[0]
        self.background = image[r1:r2, c1:c2]
        self.threshold_background = np.median(self.background[:])/15
        self.threshold_motion = np.median(self.background[:])/20
        self.stack = np.zeros((self.height, self.width, self.stack_size), dtype = np.uint8)
        for i in range(self.stack_size):
            self.stack[:,:,i] = self.background
        self.stack_count = 0
        return

    def update_background(self, crop):
        self.stack[:,:,self.stack_count] = crop
        self.stack_count = (self.stack_count + 1) % self.stack_size
        self.since_stack_update = 0
        tmp = np.uint8(np.median(self.stack, axis = 2))
        self.background = np.copy(tmp)
        return

    def add_behaviour(self, x, y, heading, area, motion):
        self.x.append(x)
        self.y.append(y)
        self.heading.append(heading)
        self.area.append(area)
        self.motion.append(motion)
        return

# FIN
