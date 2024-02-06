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
from libs.MZ_fish import Fish

# Utilities for processing videos of 96-well plate experiments

# Process Video : Make Summary Images
def process_video_summary_images(video_path, output_folder):
    
    # Load Video
    vid = cv2.VideoCapture(video_path)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read First Frame
    ret, im = vid.read()
    previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    width = np.size(previous, 1)
    height = np.size(previous, 0)
    
    # Alloctae Image Space
    bFrames = 300
    stepFrames = numFrames // bFrames
    thresholdValue=10
    accumulated_diff = np.zeros((height, width), dtype = float)
    backgroundStack = np.zeros((height, width, bFrames), dtype = float)
    background = np.zeros((height, width), dtype = float)
    bCount = 0
    for i, f in enumerate(range(0, numFrames, stepFrames)):
        
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, im = vid.read()
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        absDiff = cv2.absdiff(previous, current)
        level, threshold = cv2.threshold(absDiff,thresholdValue,255,cv2.THRESH_TOZERO)
        previous = current
       
        # Accumulate differences
        accumulated_diff = accumulated_diff + threshold
        
        # Add to background stack
        if(bCount < bFrames):
            #cv2.imwrite(output_folder + f'/current_{bCount}.png', current)
            backgroundStack[:,:,bCount] = current
            bCount = bCount + 1
        
        # Report
        print(f'{f}({bCount})')

    vid.release()

    # Normalize accumulated difference image
    accumulated_diff = accumulated_diff/np.max(accumulated_diff)
    accumulated_diff = np.ubyte(accumulated_diff*255)
    
    # Enhance Contrast (Histogram Equalize)
    equ = cv2.equalizeHist(accumulated_diff)

    # Compute Background Frame (median or mode)
    background = np.median(backgroundStack, axis = 2)

    cv2.imwrite(output_folder + r'/difference.png', equ)    
    cv2.imwrite(output_folder + r'/background.png', background)

    return

# Process Video : ROI analysis
def process_video_roi_analysis(video_path, plate, output_folder):

    # Load Video
    vid = cv2.VideoCapture(video_path)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load first frame
    ret, im = vid.read()
    previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Reset video
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Track within each ROI
    led_roi = ((0,0), (48,48))
    led_intensity = []
#    for f in range(0, numFrames):
    for f in range(0, 48000):
        
        # Read next frame        
        ret, im = vid.read()
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Compute frame-by-frame difference
        absDiff = cv2.absdiff(previous, current)
        previous = current

        # Process each fish ROI
        for fish in plate:
            # Extract Crop Region for motion estimate
            crop = get_ROI_crop(absDiff, (fish.ul, fish.lr))
            threshed = crop[crop > 10]
            motion = np.sum(threshed)
            fish.motion.append(motion)

        # Process LED
        crop = get_ROI_crop(current, led_roi)
        threshed = crop[crop > 10]
        intensity = np.sum(threshed)
        led_intensity.append(intensity)

        print(f'{numFrames-f}: {plate[44].motion[f]}')
    return plate, led_intensity


# Return cropped image from ROI list
def get_ROI_crop(image, roi):
    r1 = roi[0][1]
    r2 = roi[1][1]
    c1 = roi[0][0]
    c2 = roi[1][0]
    crop = image[r1:r2, c1:c2]
    return crop
    
# Return ROI size from ROI list
def get_ROI_size(ROIs, numROi):
    width = np.int(ROIs[numROi, 2])
    height = np.int(ROIs[numROi, 3])
    
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