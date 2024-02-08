# -*- coding: utf-8 -*-
"""
Measure behaviour in a 96-well PPI experiment

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
path_list_path = base_path + "/PPI_Behaviour/path_list.txt"
path_list = MZU.load_path_list(path_list_path)

# Anayze behaviour for video paths (*.avi) in path_list
for path in path_list:
    # Create Paths
    video_path = base_path + path
    output_folder = os.path.dirname(video_path) + '/analysis'
    figures_folder = os.path.dirname(video_path) + '/analysis/figures'
    fish_figures_folder = os.path.dirname(video_path) + '/analysis/figures/fish'
    roi_path = output_folder + '/roi.csv'
    led_path = output_folder + '/intensity.csv'
    background_path = output_folder + r'/background.png'

    # Create figures folder
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)   

    # Create fish figures folder
    if not os.path.exists(fish_figures_folder):
        os.makedirs(fish_figures_folder)   

    # Create plate structure
    plate = MZF.create_plate()

    # Load ROIs
    plate = MZR.load_rois(roi_path, plate)

    # Load LED intensity
    led_intensity = np.genfromtxt(led_path, delimiter=',')
    num_frames = len(led_intensity)

    # Extract stimulus times and types
    single_pulses, paired_pulses = MZU.extract_ppi_stimuli(led_intensity)

    # Plot LED
    fig = plt.figure(figsize=(10, 8))
    plt.title('LED-based Pulse Detection')
    for i,pulse in enumerate(single_pulses):
        plt.subplot(2,8,i+1)
        plt.plot(led_intensity[(pulse-100):(pulse+100)])
        plt.plot(100, np.max(led_intensity), 'r+')
        plt.yticks([])
    for i,pair in enumerate(paired_pulses):
        plt.subplot(2,8,i+9)
        plt.plot(led_intensity[(pair[0]-100):(pair[0]+100)])
        plt.plot(100, np.max(led_intensity), 'g+')
        plt.plot((pair[1]-pair[0])+100, np.max(led_intensity), 'b+')
        plt.yticks([])
    plt.savefig(figures_folder + '/led_intensity.png', dpi=180)
    plt.cla()
    plt.close()

    # Load fish behaviour
    plate_behaviour = np.zeros((num_frames, 5, 96), dtype=np.float32)
    fish_folder = output_folder + '/fish'
    for i, fish in enumerate(plate):
        fish_path = fish_folder + f'/{(i+1):02d}_fish.csv'
        fish_behaviour = pd.read_csv(fish_path, delimiter=",", header=None).values
        plate_behaviour[:,:,i] = fish_behaviour
        print(i)

    # Analyse
    for i, fish in enumerate(plate):
        figure_path = fish_figures_folder + f'/{(i+1):02d}_fish.png'
        x = plate_behaviour[:,0,i]
        y = plate_behaviour[:,1,i]
        area = plate_behaviour[:,2,i]
        heading = plate_behaviour[:,3,i]
        motion = plate_behaviour[:,4,i]

        fig = plt.figure(figsize=(10, 8))
        plt.subplot(2,3,1)
        plt.title('Motion')
        plt.plot(motion)
        plt.subplot(2,3,2)
        plt.title('Tracking')
        plt.plot(x,y,'.', markersize=3, color=[0,0,0,0.01])
        plt.xlim(fish.ul[0], fish.lr[0])
        plt.ylim(fish.ul[1], fish.lr[1])
        plt.subplot(2,3,3)
        plt.title('Area')
        plt.plot(area, 'm.', markersize=2, alpha=0.25)
        plt.subplot(2,3,6)
        plt.title('Heading')
        plt.plot(heading, 'g.', markersize=2, alpha=0.25)

        # PPI responses
        pre_frames = 50
        post_frames = 150
        single_responses = np.zeros((len(single_pulses), pre_frames+post_frames))
        for j, p in enumerate(single_pulses):
            single_responses[j, :] = motion[(p-(pre_frames+1)):(p+post_frames-1)]
        plt.subplot(2,3,4)
        plt.title('Single Pulse')
        plt.vlines(pre_frames, 0, np.max(single_responses), color=[0,1,0,1])
        plt.plot(single_responses.T, color=[0,0,0,0.25])
        plt.plot(np.mean(single_responses, axis=0), color=[1.0,0,0,1.0])
        plt.ylim(0, np.max(motion))
        paired_responses = np.zeros((len(paired_pulses), pre_frames+post_frames))
        for j, pair in enumerate(paired_pulses):
            p = pair[0]
            paired_responses[j, :] = motion[(p-(pre_frames+1)):(p+post_frames-1)]
        plt.subplot(2,3,5)
        plt.title('Paired Pulse')
        plt.vlines(pre_frames, 0, np.max(paired_responses), color=[0,1,0,1])
        plt.vlines(pre_frames+30, 0, np.max(paired_responses), color=[0,1,0,1])
        plt.plot(paired_responses.T, color=[0,0,0,0.25])
        plt.plot(np.mean(paired_responses, axis=0), color=[1.0,0,0,1.0])
        plt.ylim(0, np.max(motion))

        # Save
        plt.savefig(figure_path, dpi=180)
        print(i)
        plt.cla()
        plt.close()

#FIN
