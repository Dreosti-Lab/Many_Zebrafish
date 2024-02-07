# -*- coding: utf-8 -*-
"""
Many_Zebrafish: Video Library

@author: kamnpff (Adam Kampff)
"""
# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import cv2

# Import local modules
from MZ_fish import Fish

# Utilities for processing videos of 96-well plate experiments

# Process Video : Make Summary Images
def process_video_summary_images(video_path, stack_size, step_frames, output_folder):
    
    # Load Video
    vid = cv2.VideoCapture(video_path)
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read First Frame
    ret, im = vid.read()
    previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    width = np.size(previous, 1)
    height = np.size(previous, 0)
    
    # Alloctae Image Space
    if step_frames <= 0:
        step_frames = num_frames // stack_size
    threshold_value = 10
    accumulated_diff = np.zeros((height, width), dtype = float)
    background_stack = np.zeros((height, width, stack_size), dtype = float)
    background = np.zeros((height, width), dtype = float)
    stack_count = 0
    for i, f in enumerate(range(0, stack_size * step_frames, step_frames)):
        
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, im = vid.read()
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        abs_diff = cv2.absdiff(previous, current)
        level, threshold = cv2.threshold(abs_diff,threshold_value,255,cv2.THRESH_TOZERO)
        previous = current
       
        # Accumulate differences
        accumulated_diff = accumulated_diff + threshold
        
        # Add to background stack
        if(stack_count < stack_size):
            #cv2.imwrite(output_folder + f'/current_{stack_count}.png', current)
            background_stack[:,:,stack_count] = current
            stack_count = stack_count + 1
        
        # Report
        print(f'{f}({stack_count})')

    vid.release()

    # Normalize accumulated difference image
    accumulated_diff = accumulated_diff/np.max(accumulated_diff)
    accumulated_diff = np.ubyte(accumulated_diff*255)
    
    # Enhance Contrast (Histogram Equalize)
    equ = cv2.equalizeHist(accumulated_diff)

    # Compute Background Frame (median or mode)
    background = np.median(background_stack, axis = 2)

    # Store
    cv2.imwrite(output_folder + r'/difference.png', equ)    
    cv2.imwrite(output_folder + r'/background.png', background)

    return

# Process Video : ROI analysis
def process_video_roi_analysis(video_path, plate, intensity_roi, num_frames, output_folder):

    # Load Video
    vid = cv2.VideoCapture(video_path)
    if num_frames < 0:
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load first frame
    ret, im = vid.read()
    previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Reset video
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Track within each ROI
    region_intensity = []
    for f in range(0, num_frames):
        
        # Read next frame and convert to grayscale
        ret, im = vid.read()
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Compute frame-by-frame difference
        abs_diff = cv2.absdiff(previous, current)
        previous = current

        # Process each fish ROI
        for fish in plate:
            # Motion estimation
            crop = get_ROI_crop(abs_diff, (fish.ul, fish.lr))
            threshed = crop[crop > fish.threshold_motion]
            motion = np.sum(threshed)
            fish.motion.append(motion)

            # Centroid tracking
            crop = get_ROI_crop(current, (fish.ul, fish.lr))
            subtraction = cv2.subtract(fish.background, crop)
            level, threshold = cv2.threshold(subtraction,fish.threshold_background,255,cv2.THRESH_BINARY)
            threshold = np.uint8(threshold)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            contours, hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                fish.x.append(fish.ul[0])
                fish.y.append(fish.ul[1])
                fish.area.append(-1.0)
            else:
                largest_cnt, area = get_largest_contour(contours)
                if area == 0.0:
                    fish.x.append(fish.ul[0])
                    fish.y.append(fish.ul[1])
                    fish.area.append(-1.0)
                else:
                    M = cv2.moments(largest_cnt)
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    fish.x.append(fish.ul[0] + cx)
                    fish.y.append(fish.ul[1] + cy)
                    fish.area.append(area)

        # Process LED
        crop = get_ROI_crop(current, intensity_roi)
        threshed = crop[crop > 10]
        intensity = np.sum(threshed)
        region_intensity.append(intensity)

        # Report
        if (f % 1000) == 0:
            print(f'{num_frames-f}: {plate[44].motion[f]}')
    return plate, region_intensity


# Return cropped image from ROI list
def get_ROI_crop(image, roi):
    r1 = roi[0][1]
    r2 = roi[1][1]
    c1 = roi[0][0]
    c2 = roi[1][0]
    crop = image[r1:r2, c1:c2]
    return crop
    
# Return ROI size from ROI list
def get_ROI_size(ROIs, num_ROI):
    width = np.int(ROIs[num_ROI, 2])
    height = np.int(ROIs[num_ROI, 3])
    
    return width, height

# Return largest (area) cotour from contour list
def get_largest_contour(contours):
    # Find contour with maximum area and store it as best_cnt
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
    if max_area > 0:
        return best_cnt, max_area
    else:
        return cnt, max_area

# FIN
