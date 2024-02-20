# -*- coding: utf-8 -*-
"""
Many_Zebrafish: Video Library

@author: kampff
"""
# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import glob
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
def fish_tracking_roi(video_path, plate, intensity_roi, start_frame=0, end_frame=-1, max_background_rate=400, validate=False, validation_folder=None, report_interval = 1000):
    # If validating, then create validation folder
    if(validate):
        if not os.path.exists(validation_folder):
            os.makedirs(validation_folder)
    
    # Load video
    vid = cv2.VideoCapture(video_path)
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine tracking range
    if end_frame == -1:
        end_frame = start_frame + num_frames - 1
    num_frames = end_frame - start_frame + 1

    # Load first frame
    ret, im = vid.read()
    frame = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Set "previous" crops for each fish
    for fish in plate.wells:
        # Crop ROI
        crop = get_ROI_crop(frame, (fish.roi_ul, fish.roi_lr))
        fish.previous = np.copy(crop)

    # Reset video
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Video Loop
    start_time = time.time()
    for f in range(start_frame, end_frame+1, 1):

        # Is this a report/validate frame?
        report = ((f % report_interval) == 0)

        # Read next frame and convert to grayscale
        ret, im = vid.read()
        frame = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # -RV-
        if report and validate:
            display = np.copy(frame)

        # Track each fish ROI
        for fish in plate.wells:
            feedback = track_fish(frame, max_background_rate, fish)
            # -RV-
            if report and validate:
                display = set_ROI_crop(display, (fish.roi_ul, fish.roi_lr), feedback)
        
        # Process LED
        crop = get_ROI_crop(frame, intensity_roi)
        intensity = np.mean(crop)
        plate.intensity.append(intensity)

        # Validate?
        if report and validate:
            fig = plt.figure(figsize=(20, 8))
            
            # Show tracking performance
            plt.subplot(1,2,1)
            plt.imshow(im)
            for i, fish in enumerate(plate.wells):
                x = fish.x[-1]
                y = fish.y[-1]
                area = fish.area[-1]
                heading = fish.heading[-1]
                dx = math.cos((heading / 360.0) * (2 * math.pi))
                dy = -1*math.sin((heading / 360.0) * (2 * math.pi))
                if area > 0:
                    plt.plot(x,y,'go', alpha=0.25)
                    plt.plot(x + dx*10,y + dy*10,'bo', alpha=0.5, markersize=1)
                    plt.plot([x + dx*-10, x + dx*10],[y + dy*-10, y + dy*10],'y', alpha=0.2, linewidth=1)
                else:
                    plt.plot(x+fish.width/2,y+fish.height/2,'r+', alpha=0.25)

            # Show algorithm feedback
            plt.subplot(1,2,2)
            plt.imshow(display)
            validation_figure_path = validation_folder + f'/{f:010d}_frame.png'
            plt.tight_layout()
            plt.savefig(validation_figure_path, dpi=180)
            plt.close(fig)

        # Report
        if report:
            end_time = time.time()
            print(f'{f} ({end_frame}): Elapsed {end_time - start_time:.3f} s')
            start_time = time.time()
    
    # Cleanup
    vid.release()

    return plate


# Return cropped image from ROI
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

# Fish tracking algorithm
def track_fish(frame, max_background_rate, fish):
        # Crop ROI
        crop = get_ROI_crop(frame, (fish.roi_ul, fish.roi_lr))

        # Difference from background (fish always darker)
        subtraction = cv2.subtract(fish.background, crop)

        # Threshold
        level, threshold = cv2.threshold(subtraction,fish.threshold_background,255,cv2.THRESH_BINARY)

        # Morphological close
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        # If no contours, continue
        if len(contours) == 0:
            fish.add_behaviour(fish.roi_ul[0], fish.roi_ul[1], 0.0, -1.0, 0.0)
            return threshold

        # Get largest contour
        largest_cnt, area = get_largest_contour(contours)

        # If no area, continue
        if area == 0.0:
            fish.add_behaviour(fish.roi_ul[0], fish.roi_ul[1], 0.0, -1.0, 0.0)
            return threshold
        
        # Create Binary Mask Image
        mask = np.zeros(crop.shape,np.uint8)

        # Draw largest contour into Mask Image (1 for Fish, 0 for Background)
        mask = cv2.drawContours(mask,[largest_cnt],0,1,-1) # -1 draw the contour filled

        # Extract pixel points
        pixelpoints = np.transpose(np.nonzero(mask))

        # Get Area (again)
        area = np.size(pixelpoints, 0)

        # Compute Frame-by-Frame Motion (absolute intensity changes above "motion" threshold)
        motion_abs_diff = cv2.absdiff(fish.previous, crop)
        level, motion_threshold = cv2.threshold(motion_abs_diff, fish.threshold_motion, 255, cv2.THRESH_TOZERO)
        motion = np.sum(motion_threshold[:])

        # Decide whether to update the background
        if fish.since_stack_update < max_background_rate:
            fish.since_stack_update += 1
        else:
            if (motion > 500) and (fish.previous_motion > 500):
                fish.update_background(crop)

        # Update "previous" crop
        fish.previous = np.copy(crop)
        fish.previous_motion = motion

        # Extract fish pixel values (difference from background)
        r = pixelpoints[:,0]
        c = pixelpoints[:,1]
        values = subtraction[r,c].astype(float)

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
        nonzero_pts = np.sum(d, axis = 0) != 0.0
        d = d[:, nonzero_pts]
        mag = np.sqrt(np.sum(d*d, axis=0)).T
        d_norm = d/mag
        dirs = np.dot(np.array([vx, vy]), d_norm) * values[nonzero_pts]
        acc_dir = np.sum(dirs)

        # Determine heading (0 deg to right, 90 deg up)
        if acc_dir > 0:
            heading = math.atan2((-vy), (vx)) * (360.0/(2*np.pi))
        else:
            heading = math.atan2((vy), (-vx)) * (360.0/(2*np.pi))

        # Store
        fish.add_behaviour(fish.roi_ul[0] + cx, fish.roi_ul[1] + cy, heading, area, motion)

        # Return feedback
        return threshold

# Make mosaic
def make_mosaic(clip_folder, clip_size, num_rows, num_cols, num_frames, output_path):
    mosaic_width = clip_size * num_cols
    mosaic_height = clip_size * num_rows
    num_clips = num_rows * num_cols

    # Find clips
    clip_paths = glob.glob(clip_folder + '/*.avi')

    # Create mosaic
    mosaic = np.zeros((mosaic_height, mosaic_width, 3, num_frames), dtype=np.uint8)

    # Create random clip list
    clip_list = np.random.permutation(np.arange(len(clip_paths), dtype=np.int32))[:num_clips]

    # Load clips
    clips = []
    for c in clip_list:
        clip_path = clip_paths[c]
        clip = np.zeros((int(clip_size), int(clip_size), 3, int(num_frames)), dtype=np.uint8)
        vid = cv2.VideoCapture(clip_path)
        for f in range(num_frames):
            ret, frame = vid.read()
            clip[:,:,:,f] = frame
        clips.append(clip)
        vid.release()

    # Build mosaic
    count = 0
    for r in range(num_rows):
        for c in range(num_cols):
            x_off = c * clip_size
            y_off = r * clip_size
            clip = clips[count]
            mosaic[y_off:(y_off+clip_size), x_off:(x_off+clip_size), :, :] = clip
            count = count + 1
    
    # Save mosaic
    fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    video = cv2.VideoWriter(output_path, fourcc, 30, (1920,1080))
    for f in range(num_frames):
        frame = mosaic[:,:,:,f]
        resized = cv2.resize(frame, (1920,1080))
        ret = video.write(resized)
    ret = video.release()

# FIN
