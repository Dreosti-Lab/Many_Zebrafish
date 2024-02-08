# -*- coding: utf-8 -*-
"""
Validate tracking in a 96-well Sleep experiment

@author: kamnpff (Adam Kampff)
"""

# Load Environment file and variables
import os
from dotenv import load_dotenv
load_dotenv()
libs_path = os.getenv('LIBS_PATH')
base_path = os.getenv('BASE_PATH')

# Set Library Paths
import sys
sys.path.append(libs_path)

# Import useful libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2

# Import local modules
import MZ_fish as MZF
import MZ_video as MZV
import MZ_roi as MZR
import MZ_utilities as MZU

# Reload modules
import importlib
importlib.reload(MZF)
importlib.reload(MZV)
importlib.reload(MZR)
importlib.reload(MZU)

# Load list of video paths
path_list_path = base_path + "/Sleep_Behaviour/path_list.txt"
path_list = MZU.load_path_list(path_list_path)

# Validate fish tracking for video paths (*.avi) in path_list
for path in path_list:
    # Create Paths
    video_path = base_path + path
    output_folder = os.path.dirname(video_path) + '/analysis'
    fish_folder = output_folder + '/fish'
    validation_folder = output_folder + '/validation'
    roi_path = output_folder + '/roi.csv'
    intensity_path = output_folder + '/intensity.csv'
    background_path = output_folder + '/background.png'

    # Create validation folder
    if not os.path.exists(validation_folder):
        os.makedirs(validation_folder)   

    # Load Video
    vid = cv2.VideoCapture(video_path)

    # Create plate structure
    plate = MZF.create_plate()

    # Load ROIs
    plate = MZR.load_rois(roi_path, plate)

    # Load intensity
    intensity = np.genfromtxt(intensity_path, delimiter=',')
    num_frames = len(intensity)

    # Load -Initial- Background Frame
    background = cv2.imread(background_path)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # Set backgrounds
    plate = MZF.set_backgrounds(background, plate)

    # Load fish behaviour
    plate_behaviour = np.zeros((num_frames, 5, 96), dtype=np.float32)
    fish_folder = output_folder + '/fish'
    for i, fish in enumerate(plate):
        fish_path = fish_folder + f'/{(i+1):02d}_fish.csv'
        fish_behaviour = pd.read_csv(fish_path, delimiter=",", header=None).values
        plate_behaviour[:,:,i] = fish_behaviour
        print(i)
    
    # Validate
    num_validation_frames = 1000
    step_frames = num_frames // num_validation_frames
    for f in range(0, num_frames, step_frames):
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, im = vid.read()
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(im)
        for i, fish in enumerate(plate):
            x = plate_behaviour[f,0,i]
            y = plate_behaviour[f,1,i]
            area = plate_behaviour[f,2,i]
            heading = plate_behaviour[f,3,i]
            motion = plate_behaviour[f,4,i]
            dx = math.cos((heading / 360.0) * (2 * math.pi))
            dy = -1*math.sin((heading / 360.0) * (2 * math.pi))
            if area > 0:
                plt.plot(x,y,'go', alpha=0.25)
                plt.plot(x + dx*10,y + dy*10,'bo', alpha=0.5, markersize=1)
                plt.plot([x + dx*-10, x + dx*10],[y + dy*-10, y + dy*10],'y', alpha=0.2, linewidth=1)
            else:
                plt.plot(x+fish.width/2,y+fish.height/2,'r+', alpha=0.25)
        validation_figure_path = validation_folder + f'/{f:010d}_frame.png'
        plt.tight_layout()
        plt.savefig(validation_figure_path, dpi=180)
        plt.cla()
        plt.close()
        print(num_frames - f)
    
    #num_validation_frames = 100
    #step_frames = num_frames // num_validation_frames
    #for f in range(0, num_frames, step_frames):
    #    vid.set(cv2.CAP_PROP_POS_FRAMES, f)
    #    ret, im = vid.read()
    #    current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #    fig = plt.figure(figsize=(20, 8))
    #    plt.subplot(1,2,1)
    #    plt.imshow(im)
    #    for i, fish in enumerate(plate):
    #        x = plate_behaviour[f,1,i]
    #        y = plate_behaviour[f,2,i]
    #        area = plate_behaviour[f,3,i]
    #        if area > 0:
    #            plt.plot(x,y,'go', alpha=0.25)
    #        else:
    #            plt.plot(x+fish.width/2,y+fish.height/2,'r+', alpha=0.25)
    #    plt.subplot(1,2,2)
    #    display = np.copy(current)
    #    for i, fish in enumerate(plate):
    #        crop = MZV.get_ROI_crop(current, (fish.ul, fish.lr))
    #        subtraction = cv2.subtract(fish.background, crop)
    #        level, threshold = cv2.threshold(subtraction,fish.threshold_background,255,cv2.THRESH_BINARY)
    #        threshold = np.uint8(threshold)
    #        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #        closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    #        contours, hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #        if len(contours) != 0:
    #            largest_cnt, area = MZV.get_largest_contour(contours)
    #            if area > 0.0:
    #                M = cv2.moments(largest_cnt)
    #                cx = M["m10"] / M["m00"]
    #                cy = M["m01"] / M["m00"]
    #                mask = np.zeros(crop.shape,np.uint8)
    #                mask = cv2.drawContours(mask,[largest_cnt],0,255,-1) # -1 draw the contour filled
    #        display = MZV.set_ROI_crop(display, (fish.ul, fish.lr), mask)
    #    plt.imshow(display)
    #    validation_figure_path = validation_folder + f'/{f:010d}_frame.png'
    #    plt.savefig(validation_figure_path, dpi=180)
    #    plt.cla()
    #    plt.close()
#
    #    fish = plate[69]
    #    fig = plt.figure(figsize=(10, 8))
    #    plt.subplot(2,3,1)
    #    crop = MZV.get_ROI_crop(current, (fish.ul, fish.lr))
    #    plt.imshow(crop)
    #    subtraction = cv2.subtract(fish.background, crop)
    #    plt.subplot(2,3,2)
    #    plt.imshow(subtraction)
    #    level, threshold = cv2.threshold(subtraction,fish.threshold_background,255,cv2.THRESH_BINARY)
    #    plt.subplot(2,3,3)
    #    plt.imshow(threshold)
    #    threshold = np.uint8(threshold)
    #    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    #    plt.subplot(2,3,4)
    #    plt.imshow(closing)
    #    contours, hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #    cx = 0
    #    cy = 0
    #    mask = np.zeros(crop.shape,np.uint8)
    #    if len(contours) != 0:
    #        largest_cnt, area = MZV.get_largest_contour(contours)
    #        if area > 0.0:
    #            M = cv2.moments(largest_cnt)
    #            cx = M["m10"] / M["m00"]
    #            cy = M["m01"] / M["m00"]
    #            mask = cv2.drawContours(mask,[largest_cnt],0,255,-1) # -1 draw the contour filled
    #    plt.subplot(2,3,5)
    #    plt.imshow(mask)
    #    plt.plot(cx,cy, 'r+')
    #    tracking_figure_path = validation_folder + f'/{f:010d}_tracking.png'
    #    plt.savefig(tracking_figure_path, dpi=180)
    #    plt.cla()
    #    plt.close()
#
        print(num_frames - f)

    # Cleanup
    vid.release()

#FIN
