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

# Import modules
import MZ_plate as MZP
import MZ_video as MZV

# Reload modules
import importlib
importlib.reload(MZP)
importlib.reload(MZV)

# Utilities for analysing and inspecting bouts performed in 96-well plate experiments

# Outlier interpolation 
def interpolate_outliers(array, outliers):
    for outlier in outliers:
        start_index = outlier[0]
        stop_index = outlier[1]
        start_value = array[start_index-1]
        stop_value = array[stop_index]
        delta = stop_value-start_value
        for j, index in enumerate(range(start_index, stop_index)):
            array[index] = start_value + (j * delta)
    return array

# Outlier removal (centroid)
def remove_outliers(bout, window_size, threshold):
    smoothed = np.copy(bout)
    x = bout[:, 0]
    y = bout[:, 1]
    heading = bout[:, 2]
    area = bout[:, 3]
    bout_length = len(x)
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    x_overs = dx > threshold
    x_unders = dx < -threshold
    y_overs = dy > threshold
    y_unders = dy < -threshold
    overs = x_overs
    unders = x_unders
    outliers = []
    for i in range(bout_length-window_size):
        over_window = overs[i:(i+window_size)]
        under_window = unders[i:(i+window_size)]
        num_overs = np.sum(over_window)
        num_unders = np.sum(under_window)
        if((num_overs == 1) and (num_unders == 1)):
            over_index = np.where(over_window)[0][0]
            under_index = np.where(under_window)[0][0]
            overs[i+over_index] = False
            unders[i+under_index] = False
            if(over_index > under_index):
                first = i+under_index
                second = i+over_index
            else:
                first = i+over_index
                second = i+under_index
            print(f"outliers: {first} and {second} ({i})")
            outliers.append((first, second))
    if len(outliers) > 0:
        smoothed[:, 0] = interpolate_outliers(x, outliers)
        smoothed[:, 1] = interpolate_outliers(y, outliers)
    overs = y_overs
    unders = y_unders
    outliers = []
    for i in range(bout_length-window_size):
        over_window = overs[i:(i+window_size)]
        under_window = unders[i:(i+window_size)]
        num_overs = np.sum(over_window)
        num_unders = np.sum(under_window)
        if((num_overs == 1) and (num_unders == 1)):
            over_index = np.where(over_window)[0][0]
            under_index = np.where(under_window)[0][0]
            overs[i+over_index] = False
            unders[i+under_index] = False
            if(over_index > under_index):
                first = i+under_index
                second = i+over_index
            else:
                first = i+over_index
                second = i+under_index
            print(f"outliers: {first} and {second} ({i})")
            outliers.append((first, second))
    if len(outliers) > 0:
        smoothed[:, 0] = interpolate_outliers(x, outliers)
        smoothed[:, 1] = interpolate_outliers(y, outliers)
    return smoothed

# Smooth bout (remove obvious tracking failures)
def smooth_bout(bout):
    copy = np.copy(bout)
    smoothed = remove_outliers(copy, 10, 5)
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(bout[:, i])
        plt.plot(smoothed[:, i])
    plt.show()
    return smoothed

# Is this bout valid?
#  < 5% tracking failures
def is_valid_bout(bout):
    bout_length = bout.shape[0]
    x = bout[:, 0]
    y = bout[:, 1]
    area = bout[:, 3]
    invalid_frames = np.sum(area == -1.0)
    if invalid_frames > (0.10 * bout_length):
        return False

    # Are there tracking jumps (big x/y shifts that flip)?
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    return True

# Accumulate angle (heading) (and remove flips and zero crossings)
def cumulative_angle(heading):
    delta = np.diff(heading, prepend=heading[0])
    # Find zero crossings and correct
    crossings = np.abs(delta) > 270
    delta[crossings] = delta[crossings] - (np.sign(delta[crossings]) * 360.0)
    # Find flips (heading swap errors) and zero
    flips = np.abs(delta) > 160
    num_flips = np.sum(flips)
    delta[flips] = 0
    cumulative_angle = np.cumsum(delta)
    return num_flips, cumulative_angle

# Measure bout
def measure_response(bout, stimulus_index, motion_threshold):
    bout_length = bout.shape[0]
    x = bout[:, 0]
    y = bout[:, 1]
    heading = bout[:, 2]
    num_flips, cumulative_heading = cumulative_angle(heading)
    plt.plot(cumulative_heading)
    print(num_flips)
    plt.show()
    area = bout[:, 3]
    motion = bout[:, 4]
    start_index = stimulus_index + np.where(motion[stimulus_index:-1] > motion_threshold)[0][0]
    stop_index = start_index + np.where(motion[start_index:-1] < motion_threshold)[0][0]
    duration = stop_index-start_index
    latency = start_index - stimulus_index
    start_x = np.median(x[(start_index-6):(start_index-1)])
    start_y = np.median(y[(start_index-6):(start_index-1)])
    start_heading = np.median(cumulative_heading[(start_index-6):(start_index-1)])
    stop_x = np.median(x[stop_index:(stop_index+5)])
    stop_y = np.median(y[stop_index:(stop_index+5)])
    stop_heading = np.median(cumulative_heading[stop_index:(stop_index+5)])
    dx = stop_x - start_x
    dy = stop_y - start_y
    distance = np.sqrt((dx*dx) + (dy*dy))
    turning = stop_heading - start_heading
    return (duration, latency, distance, turning)

# Inspect bout
def inspect_bout(movie, roi, bout, clip_path):
    valid_bout = is_valid_bout(bout)
    metrics = measure_response(bout, 51, 20)
    signal = bout[:,4]
    bout_length = bout.shape[0]
    clip_size = 256
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter(clip_path, fourcc, 30, (clip_size,clip_size))
    for frame_index in range(0, bout_length):
        frame = movie[frame_index]
        crop = MZV.get_ROI_crop(frame, roi)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (clip_size, clip_size))
        enhanced = cv2.normalize(resized, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        x = bout[frame_index, 0]
        y = bout[frame_index, 1]
        heading = bout[frame_index, 2]
        motion = bout[frame_index, 4]
        offset = (roi[0][0],roi[0][1])
        scale = ((clip_size/crop.shape[1]), (clip_size/crop.shape[0]))
        rgb = plot_signal(rgb, signal, 2, 0.01, (255,0,255), 1, highlight=frame_index)
        rgb = draw_trajectory(rgb, bout, offset, scale, 1, (0,255,0), 1)
        rgb = plot_fish(rgb, (x,y), heading, offset, scale, (255,0,0), 1)
        rgb = draw_response_type(rgb, valid_bout)
        rgb = draw_response_metrics(rgb, metrics)
        ret = video.write(rgb)
    ret = video.release()
    return

# Draw bout trajectory
def draw_trajectory(image, bout, offset, scale, radius, color, thickness):
    bout_length = bout.shape[0]
    x = (bout[:, 0] - offset[0]) * scale[0]
    y = (bout[:, 1] - offset[1]) * scale[1]
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
    image = cv2.circle(image, (cx, cy), 2, (0, 255, 255), 1)
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 1
    pos = (15,55)
    thickness = 2
    line_type = 1
    if valid_response:
        color = (0,255,0)
        image = cv2.putText(image,'O', pos, font, size, color, thickness, line_type)
    else:
        color = (0,0,255)
        image = cv2.putText(image,'X', pos, font, size, color, thickness, line_type)
    return image

# Draw response metrics
def draw_response_metrics(image, metrics):
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 1
    pos = (5,25)
    thickness = 1
    line_type = 1
    color = (0,128,255)
    metric_text = f'{metrics[0]:01d} {metrics[1]:01d} {int(metrics[2]):02d} {int(metrics[3]):03d}'
    image = cv2.putText(image, metric_text, pos, font, size, color, thickness, line_type)
    return image

#FIN