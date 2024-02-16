# -*- coding: utf-8 -*-
"""
Many_Zebrafish: Bouts Library

@author: kampff
"""
# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

# Utilities for analysing bouts performed in 96-well plate experiments

# Is this bout valid?
#  - < 5% tracking failures
def valid_bout(bout):
    bout_length = bout.shape[1]
    area = bout[3, :]
    invalid_frames = np.sum(area == -1.0)
    if invalid_frames > (0.05 * bout_length):
        return False
    else:
        return True

# Draw bout trajectory
def draw_trajectory(image, bout, offset, scale, radius, color, thickness):
    bout_length = bout.shape[1]
    x = (bout[0, :] - offset[0]) * scale[0]
    y = (bout[1, :] - offset[1]) * scale[1]
    for i in range(bout_length):
        cx = int(round(x[i]))
        cy = int(round(y[i]))
        image = cv2.circle(image, (cx, cy), radius, color, thickness)
    return image

# Plot line
def plot_line(image, bout, offset, scale, line_color, line_thickness):
    bout_length = bout.shape[1]
    x = (bout[0, :] - offset[0]) * scale[0]
    y = (bout[1, :] - offset[1]) * scale[1]
    for i in range(bout_length -1):
        x1 = int(round(x[i]))
        y1 = int(round(y[i]))
        x2 = int(round(x[i+1]))
        y2 = int(round(y[i+1]))
        image = cv2.line(image, (x1, y1), (x2, y2), line_color, line_thickness)
    return image

#FIN