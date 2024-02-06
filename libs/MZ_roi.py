# -*- coding: utf-8 -*-
"""
Many_Zebrafish: ROI Library

@author: kamnpff (Adam Kampff)
"""
# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import cv2
from skimage.measure import profile_line

# Import local modules
from libs.MZ_fish import Fish

# Utilities for extracting ROIs from 96-well plate experiments
    
# Find ROIs
def find_rois(image_path, plate, output_folder):
    # Load image and convert to grayscale
    image  = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find extremes
    start_x, end_x, start_y, end_y = find_extremes(gray)
    arena_width = end_x - start_x
    arena_height = end_y - start_y
    roi_width = arena_width / 12.0
    roi_height = arena_height / 8.0

    # Create display image and draw arena
    display = np.copy(image)
    display = cv2.rectangle(display, (start_x, start_y), (end_x, end_y), (0,255,255), 3)

    # Find best internal dividing vertical lines
    vert_dividers = [((start_x, start_y),(start_x, end_y))]
    for col in range(1,12):
        internal_x = int(start_x + (col * roi_width))
        x1_range = np.arange(internal_x-10, internal_x+10)
        y1_range = np.array([start_y])
        x2_range = np.arange(internal_x-10, internal_x+10)
        y2_range = np.array([end_y])
        start, end = minimal_line(gray, x1_range, y1_range, x2_range, y2_range)
        vert_dividers.append((start,end))
        print(f'Verts: {col}: {start},{end}')
        display = cv2.line(display, (start), (end), (0,0,255), 1)
    vert_dividers.append(((end_x, start_y),(end_x, end_y)))

    # Find best internal dividing horizontal lines
    horz_dividers = [((start_x, start_y),(end_x, start_y))]
    for row in range(1,8):
        internal_y = int(start_y + (row * roi_height))
        x1_range = np.array([start_x])
        y1_range = np.arange(internal_y-10, internal_y+10)
        x2_range = np.array([end_x])
        y2_range = np.arange(internal_y-10, internal_y+10)
        start, end = minimal_line(gray, x1_range, y1_range, x2_range, y2_range)
        horz_dividers.append((start,end))
        print(f'Horzs: {row}: {start},{end}')
        display = cv2.line(display, (start), (end), (0,0,255), 1)
    horz_dividers.append(((start_x, end_y),(end_x, end_y)))

    # Find intersection points and create ROIs
    count = 0
    for row in range(8):
        for col in range(12):
            top = horz_dividers[row]
            bottom = horz_dividers[row+1]
            left = vert_dividers[col]
            right = vert_dividers[col+1]
            ul = find_intersection(left, top)
            lr = find_intersection(right, bottom)
            plate[count].set_roi(ul, lr)
            count = count + 1

    # Draw ROI intersections
    for fish in plate:
        display = cv2.circle(display, fish.ul, 3, (0,255,0), 1)
        display = cv2.circle(display, fish.lr, 3, (0,255,0), 1)

    # Store ROI image
    ret = cv2.imwrite(output_folder + r'/roi.png', display)

    # Store ROIs (positions)
    roi_path = output_folder + '/roi.csv'
    roi_file = open(roi_path, 'w')
    for fish in plate:
        ret = roi_file.write(f'{fish.ul[0]},{fish.ul[1]},{fish.lr[0]},{fish.lr[1]}\n')

    return plate

# Load ROIs
def load_rois(roi_path, plate):
    roi_data = np.genfromtxt(roi_path, delimiter=',')
    for i in range(96):
        roi_line = roi_data[i]
        ul = (roi_line[0],roi_line[1])
        lr = (roi_line[2],roi_line[3])
        plate[i].set_roi(ul, lr)
    return plate

# Find extremes
def find_extremes(image):
    threshold = 3
    width = image.shape[1]
    height = image.shape[0]
    vert_projection = np.mean(image, axis=0)
    horz_projection = np.mean(image, axis=1)
    
    # Find extremes
    count = 0
    for i, p in enumerate(vert_projection):
        if p >= threshold:
            count = count + 1
            if count >= 3:
                start_x = i-3
                break
        else:
            count = 0
    count = 0
    for i, p in enumerate(reversed(vert_projection)):
        if p >= threshold:
            count = count + 1
            if count >= 3:
                end_x = width-i+3
                break
        else:
            count = 0
    count = 0
    for i, p in enumerate(horz_projection):
        if p >= threshold:
            count = count + 1
            if count >= 3:
                start_y = i-3
                break
        else:
            count = 0
    count = 0
    for i, p in enumerate(reversed(horz_projection)):
        if p >= threshold:
            count = count + 1
            if count >= 3:
                end_y = height-i+3
                break
        else:
            count = 0
    return start_x, end_x, start_y, end_y


# Find minimal line (brute force)
def minimal_line(image, x1_range, y1_range, x2_range, y2_range):
    minima = 9999999999
    count = 0
    for x1 in x1_range:
        for y1 in y1_range:
            for x2 in x2_range:
                for y2 in y2_range:
                    start = (x1,y1)
                    end = (x2,y2)
                    intensity = line_intensity(image, start, end)
                    count = count + 1
                    #print(f'{y1}, {y2}: {intensity} - {count}')
                    if intensity < minima:
                        best_start = start
                        best_end = end
                        minima = intensity
    return best_start, best_end

# Compute intensity along a line
def line_intensity(image, start_xy, end_xy):
    start_rc = start_xy[::-1]
    end_rc = end_xy[::-1]
    pixels = profile_line(image, start_rc, end_rc)
    return np.mean(pixels)

# Find intersection of two lines
def find_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)

# FIN
