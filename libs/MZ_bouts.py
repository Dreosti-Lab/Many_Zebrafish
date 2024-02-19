# -*- coding: utf-8 -*-
"""
Many_Zebrafish: Bouts Library

@author: kampff
"""
# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
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

# Extract bouts
def extract_bouts(behaviour):
    upper_threshold = 250
    lower_threshold = 150
    x = behaviour[:, 0]
    y = behaviour[:, 1]
    heading = behaviour[:, 2]
    cumulative_heading = cumulative_angle(heading)
    area = behaviour[:, 3]
    motion = behaviour[:, 4]
    bout_filter = np.array([0.25, 0.25, 0.25, 0.25])
    bout_signal = signal.fftconvolve(motion, bout_filter, 'same')
    starts, peaks, stops = find_peaks_dual_threshold(bout_signal, upper_threshold, lower_threshold)
    num_bouts = len(starts)
    bouts = []
    for b in range(num_bouts):
        start = starts[b]
        peak = peaks[b]
        stop = stops[b]
        duration = stop-start
        max = bout_signal[peak]
        magnitude = np.sum(bout_signal[start:stop])
        start_x = np.median(x[(start-6):(start-1)])
        start_y = np.median(y[(start-6):(start-1)])
        start_heading = np.median(cumulative_heading[(start-6):(start-1)])
        stop_x = np.median(x[stop:(stop+5)])
        stop_y = np.median(y[stop:(stop+5)])
        stop_heading = np.median(cumulative_heading[stop:(stop+5)])
        dx = stop_x - start_x
        dy = stop_y - start_y
        distance = np.sqrt((dx*dx) + (dy*dy))
        turning = stop_heading - start_heading
        bouts.append({  'start':start, 
                        'peak':peak, 
                        'stop':stop,
                        'duration':duration,
                        'max':max,
                        'magnitude':magnitude,
                        'distance':distance,
                        'turning':turning   })
    return bouts

# Is this response valid?
#  < 5% tracking failures (area = -1.0)
#  < 2% heading flips
def validate_response(response):
    response_length = response.shape[0]
    x = response[:, 0]
    y = response[:, 1]
    heading = response[:, 2]
    area = response[:, 3]
    invalid_frames = np.sum(area == -1.0)
    if invalid_frames > (0.05 * response_length):
        return False
    num_flips = count_heading_flips(heading)
    if num_flips > (0.02 * response_length):
        return False  
    return True

# Is this response a response to the stimulus?
def classify_response(metrics):
    if (metrics['distance'] > 5) or (np.abs(metrics['turning']) > 15):
        if(metrics['latency'] < 10):
            return True
    return False

# Count heading flips
def count_heading_flips(heading):
    delta = np.diff(heading, prepend=heading[0])
    # Find zero crossings and correct
    crossings = np.abs(delta) > 270
    delta[crossings] = delta[crossings] - (np.sign(delta[crossings]) * 360.0)
    # Find flips (heading swap errors)
    flips = np.abs(delta) > 150
    num_flips = np.sum(flips)
    return num_flips

# Accumulate heading angle (and remove zero crossings)
def cumulative_angle(heading):
    delta = np.diff(heading, prepend=heading[0])
    # Find zero crossings and correct
    crossings = np.abs(delta) > 270
    delta[crossings] = delta[crossings] - (np.sign(delta[crossings]) * 360.0)
    # Find flips and zero (heading swap errors)
    flips = np.abs(delta) > 150
    delta[flips] = 0.0
    cumulative_angle = np.cumsum(delta)
    return cumulative_angle

# Measure response to simulus
def measure_response(response, stimulus_index):
    bouts = extract_bouts(response)
    # Find first bout after stimulus
    first_bout = None
    for bout in bouts:
        if bout['start'] >= stimulus_index:
            first_bout = bout
            break
    if(first_bout is not None):
        latency = first_bout['start'] - stimulus_index
        magnitude = first_bout['magnitude']
        distance = first_bout['distance']
        turning = first_bout['turning']
    else:
        magnitude = 0
        latency = 0
        distance = 0
        turning = 0
    return {'latency':latency, 'magnitude':magnitude, 'distance':distance, 'turning':turning}

# Inspect response
def inspect_response(movie, roi, response, clip_path):
    is_valid = validate_response(response)
    metrics = measure_response(response, 50)
    is_response = classify_response(metrics)
    signal = response[:,4]
    response_length = response.shape[0]
    clip_size = 256
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter(clip_path, fourcc, 30, (clip_size,clip_size))
    for frame_index in range(0, response_length):
        frame = movie[frame_index]
        crop = MZV.get_ROI_crop(frame, roi)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (clip_size, clip_size))
        enhanced = cv2.normalize(resized, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        x = response[frame_index, 0]
        y = response[frame_index, 1]
        heading = response[frame_index, 2]
        motion = response[frame_index, 4]
        offset = (roi[0][0],roi[0][1])
        scale = ((clip_size/crop.shape[1]), (clip_size/crop.shape[0]))
        rgb = plot_signal(rgb, signal, 2, 0.01, (255,0,255), 1, highlight=frame_index)
        rgb = draw_response_trajectory(rgb, response, offset, scale, 1, (0,255,0), 1)
        rgb = draw_fish(rgb, (x,y), heading, offset, scale, (255,0,0), 1)
        rgb = draw_response_type(rgb, is_valid, is_response)
        rgb = draw_response_metrics(rgb, metrics)
        ret = video.write(rgb)
    ret = video.release()
    return

# Draw response trajectory
def draw_response_trajectory(image, response, offset, scale, radius, color, thickness):
    response_length = response.shape[0]
    x = (response[:, 0] - offset[0]) * scale[0]
    y = (response[:, 1] - offset[1]) * scale[1]
    for i in range(response_length):
        cx = int(round(x[i]))
        cy = int(round(y[i]))
        image = cv2.circle(image, (cx, cy), radius, color, thickness)
    return image

# Plot "fish" - heading + centroid
def draw_fish(image, centroid, heading, offset, scale, line_color, line_thickness):
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
def draw_response_type(image, is_valid, is_response):
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 1
    pos = (15,55)
    thickness = 2
    line_type = 1
    if is_valid:
        color = (0,255,0)
    else:
        color = (0,0,255)
    if is_response:
        image = cv2.putText(image,'O', pos, font, size, color, thickness, line_type)
    else:
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
    latency = metrics['latency']
    magnitude = metrics['magnitude']
    distance = metrics['distance']
    turning = metrics['turning']
    metric_text = f'{latency:01d} {int(magnitude//100):02d} {int(distance):02d} {int(turning):+03d}'
    image = cv2.putText(image, metric_text, pos, font, size, color, thickness, line_type)
    return image

# Peak Detection
def find_peaks_dual_threshold(values, upper_threshold, lower_threshold):    
    over = 0
    starts = []
    peaks = []
    stops = []
    cur_peak_val = 0
    cur_peak_idx = 0
    num_samples = values.shape[0]
    steps = range(num_samples)
    for i in steps[6:-5]:
        if over == 0:
            if values[i] > upper_threshold:
                over = 1
                cur_peak_val = values[i]
                cur_peak_idx = i                                
                starts.append(i)
        else: # This is what happens when over the upper_threshold
            if values[i] > cur_peak_val:
                cur_peak_val = values[i]
                cur_peak_idx = i
            elif values[i] < lower_threshold:
                over = 0
                cur_peak_val = 0
                peaks.append(cur_peak_idx)
                stops.append(i)
    if(len(starts) > len(stops)):
        stops.append(num_samples-1)
        peaks.append(cur_peak_idx)
    return starts, peaks, stops

#FIN