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
import time
import cv2

# Import local modules
from MZ_fish import Fish

# Utilities for processing videos of 96-well plate experiments

# Generate initial background
def generate_initial_background(video_path, stack_size, step_frames, output_folder):
    
    # Load video
    vid = cv2.VideoCapture(video_path)
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    # Allocate stack space
    if step_frames <= 0:
        step_frames = num_frames // stack_size
    threshold_value = 10
    background_stack = np.zeros((frame_height, frame_width, stack_size), dtype = np.uint8)
    background = np.zeros((frame_height, frame_width), dtype = float)
    stack_count = 0
    for i, f in enumerate(range(0, stack_size * step_frames, step_frames)):
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, im = vid.read()
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        # Add to background stack
        if(stack_count < stack_size):
            background_stack[:,:,stack_count] = current
            stack_count = stack_count + 1
        
        # Report
        print(f'{f}({stack_count})')

    vid.release()

    # Compute Background Frame (median or mode)
    background = np.uint8(np.median(background_stack, axis = 2))

    # Store
    cv2.imwrite(output_folder + r'/background.png', background)

    # Cleanup
    vid.release()

    return

# Generate difference image
def generate_difference_image(video_path, stack_size, step_frames, output_folder):
    
    # Load video
    vid = cv2.VideoCapture(video_path)
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read first frame
    ret, im = vid.read()
    previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Allocate accumulator space
    accumulated_diff = np.zeros((frame_height, frame_width), dtype = float)

    # Set step size
    if step_frames <= 0:
        step_frames = num_frames // stack_size

    # Accumulate frame-by-frame difference values (above threshold)
    threshold_value = 10
    for i, f in enumerate(range(0, stack_size * step_frames, step_frames)):
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, im = vid.read()
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        abs_diff = cv2.absdiff(previous, current)
        level, threshold = cv2.threshold(abs_diff,threshold_value,255,cv2.THRESH_TOZERO)
        previous = current
       
        # Accumulate differences
        accumulated_diff = accumulated_diff + threshold
                
        # Report
        print(f'{f} of {num_frames}')

    vid.release()

    # Normalize accumulated difference image
    accumulated_diff = accumulated_diff/np.max(accumulated_diff)
    accumulated_diff = np.ubyte(accumulated_diff*255)
    
    # Enhance Contrast (Histogram Equalize)
    equ = cv2.equalizeHist(accumulated_diff)

    # Store
    cv2.imwrite(output_folder + r'/difference.png', equ)    

    # Cleanup
    vid.release()

    return

# Track fish within ROIs
def fish_tracking_roi(video_path, plate, intensity_roi, num_frames, max_background_rate, output_folder):

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
    
    # Video Loop
    region_intensity = []
    report_interval = 100
    start_time = time.time()
    for f in range(0, num_frames):
        
        # Read next frame and convert to grayscale
        ret, im = vid.read()
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Compute frame-by-frame difference
        abs_diff = cv2.absdiff(previous, current)
        previous = current

        # Track each fish ROI
        for fish in plate:
            # Crop ROI
            crop = get_ROI_crop(current, (fish.ul, fish.lr))

            # Absolute difference from background
            abs_diff = cv2.absdiff(fish.background, crop)

            # Threshold
            level, threshold = cv2.threshold(abs_diff,fish.threshold_background,255,cv2.THRESH_BINARY)

            # Morphological close
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

            # If no contours, continue
            if len(contours) == 0:
                fish.add_behaviour(fish.ul[0], fish.ul[1], 0.0, -1.0, 0.0)
                continue

            # Get largest contour
            largest_cnt, area = get_largest_contour(contours)

            # If no area, continue
            if area == 0.0:
                fish.add_behaviour(fish.ul[0], fish.ul[1], 0.0, -1.0, 0.0)
                continue
            
            # Create Binary Mask Image
            mask = np.zeros(crop.shape,np.uint8)

            # Draw largest contour into Mask Image (1 for Fish, 0 for Background)
            cv2.drawContours(mask,[largest_cnt],0,1,-1) # -1 draw the contour filled

            # Extract pixel points
            pixelpoints = np.transpose(np.nonzero(mask))

             # Get Area (again)
            area = np.size(pixelpoints, 0)

            # ---------------------------------------------------------------------------------
            # Compute Frame-by-Frame Motion (absolute changes above threshold)
            # - Normalize by total abs(differece) from background
            if (f != 0):
                # Measure total absolute intensity difference from background (above threshold)
                abs_diff[abs_diff < fish.threshold_motion] = 0
                total_abs_diff = np.sum(np.abs(abs_diff))
                
                # Measure frame-by-frame absolute intensity difference and normalize
                frame_by_frame_abs_diff = np.abs(np.float32(fish.previous) - np.float32(crop)) / 2 # Adjust for increases and decreases across frames
                frame_by_frame_abs_diff[frame_by_frame_abs_diff < fish.threshold_motion] = 0
                if (total_abs_diff != 0) and len(frame_by_frame_abs_diff != 0):
                    motion = np.sum(np.abs(frame_by_frame_abs_diff))/total_abs_diff
                else:
                    motion = 0.0
            else:
                motion = 0.0
            # ---------------------------------------------------------------------------------
            # Decide whether to update the background
            if fish.frames_since_background_update < max_background_rate:
                fish.frames_since_background_update += 1
            else:
                if motion > 0.1:
                    fish.update_background(crop)

            # Update "previous" crop
            fish.previous = np.copy(crop)

            # Extract fish pixel values (difference from background)
            r = pixelpoints[:,0]
            c = pixelpoints[:,1]
            values = abs_diff[r,c].astype(float)

            # Compute centroid
            r = r.astype(float)
            c = c.astype(float)
            acc = np.sum(values)
            cx = np.float32(np.sum(c*values))/acc
            cy = np.float32(np.sum(r*values))/acc

            # Compute orientation
            line = cv2.fitLine(pixelpoints, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
            vx = line[1][0]
            vy = line[0][0]

            # Score points
            dx = c - cx
            dy = r - cy
            d = np.vstack((dx, dy))
            d_norm = d/np.sqrt(np.sum(d*d, axis=0)).T
            dirs = np.dot(np.array([vx, vy]), d_norm) * values
            acc_dir = np.sum(dirs)

            # Determine heading (0 deg to right, 90 deg up)
            if acc_dir > 0:
                heading = math.atan2((-vy), (vx)) * (360.0/(2*np.pi))
            else:
                heading = math.atan2((vy), (-vx)) * (360.0/(2*np.pi))

            # Store
            fish.add_behaviour(fish.ul[0] + cx, fish.ul[1] + cy, heading, area, motion)
        
        # Process LED
        crop = get_ROI_crop(current, intensity_roi)
        threshed = crop[crop > 10]
        intensity = np.sum(threshed)
        region_intensity.append(intensity)

        # Report
        if ((f % report_interval) == 0) and (f != 0):
            end_time = time.time()
            print(f'{num_frames-f}: Elapsed {end_time - start_time:.3f} s')
            start_time = time.time()
            cv2.imwrite(output_folder + f'/debug/{f:08d}_background.png', fish.background)
            crop = get_ROI_crop(current, (plate[95].ul, plate[95].lr))
            cv2.imwrite(output_folder + f'/debug/{f:08d}_crop.png', crop)

    # Cleanup
    vid.release()

    return plate, region_intensity


# Return cropped image from ROI list
def get_ROI_crop(image, roi):
    r1 = roi[0][1]
    r2 = roi[1][1]
    c1 = roi[0][0]
    c2 = roi[1][0]
    crop = image[r1:r2, c1:c2]
    return crop
    
# Set ROI image to crop values
def set_ROI_crop(image, roi, crop):
    r1 = roi[0][1]
    r2 = roi[1][1]
    c1 = roi[0][0]
    c2 = roi[1][0]
    image[r1:r2, c1:c2] = crop
    return image

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
