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
    if invalid_frames > (0.10 * bout_length):
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

# Plot "fish" - heading + centroid
def plot_fish(image, centroid, heading, offset, scale, line_color, line_thickness):
    theta = (heading / 360.0) * 2.0 * np.pi
    dx = np.cos(theta)
    dy = -np.sin(theta)
    cx = int(round((centroid[0] - offset[0]) * scale[0]))
    cy = int(round((centroid[1] - offset[1]) * scale[1]))
    hx = cx + int(round(25 * dx))
    hy = cy + int(round(25 * dy))
    tx = cx + int(round(25 * -dx))
    ty = cy + int(round(25 * -dy))
    image = cv2.line(image, (hx, hy), (tx, ty), line_color, line_thickness)
    image = cv2.circle(image, (hx, hy), 3, (255, 255, 0), 1)
    image = cv2.circle(image, (cx, cy), 3, (0, 255, 255), 2)
    return image

# Plot signal
def plot_signal(image, signal, vertical_offset, vertical_scale, line_color, line_thickness, highlight=-1):
    array_length = signal.shape[0]
    width = image.shape[1]
    height = image.shape[1]
    x = np.arange(0, width, width/array_length)
    y = height - ((signal  * vertical_scale) + vertical_offset)
    for i in range(array_length-1):
        x1 = int(round(x[i]))
        y1 = int(round(y[i]))
        x2 = int(round(x[i+1]))
        y2 = int(round(y[i+1]))
        image = cv2.line(image, (x1, y1), (x2, y2), line_color, line_thickness)
    if highlight >= 0:
        cx = int(round(x[highlight]))
        cy = int(round(y[highlight]))
        image = cv2.circle(image, (cx, cy), 2, (0, 255, 255), 1)
    return image

# Draw response type
def draw_response_type(image, valid_response):
    width = image.shape[1]
    height = image.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 2
    pos = (int(width/2)-size*10,int(height/2)+size*10)
    thickness = 3
    line_type = 2
    if valid_response:
        color = (0,255,0)
        image = cv2.putText(image,'O', pos, font, size, color, thickness, line_type)
    else:
        color = (0,0,255)
        image = cv2.putText(image,'X', pos, font, size, color, thickness, line_type)
    return image

#FIN