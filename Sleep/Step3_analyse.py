# -*- coding: utf-8 -*-
"""
Measure behaviour in a 96-well Sleep experiment

@author: kampff
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
path_list_path = base_path + "/Sleep_Behaviour/path_list.txt"
path_list = MZU.load_path_list(path_list_path)

# Anayze behaviour for video paths (*.avi) in path_list
for path in path_list:
    # Create Paths
    video_path = base_path + path
    output_folder = os.path.dirname(video_path) + '/analysis'
    figures_folder = os.path.dirname(video_path) + '/analysis/figures'
    fish_figures_folder = os.path.dirname(video_path) + '/analysis/figures/fish'
    roi_path = output_folder + '/roi.csv'
    intensity_path = output_folder + '/intensity.csv'
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

    # Load intensity
    intensity = np.genfromtxt(intensity_path, delimiter=',')
    num_frames = len(intensity)

    # Plot intensity
    fig = plt.figure(figsize=(10, 4))
    plt.title('Background Intensity Detection')
    plt.plot(intensity)
    plt.savefig(figures_folder + '/intensity.png', dpi=180)
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
        plt.subplot(2,2,1)
        plt.title('Motion')
        plt.plot(motion)
        plt.ylim(0, 10000)
        plt.subplot(2,2,2)
        plt.title('Tracking')
        plt.plot(x,y,'.', markersize=3, color=[0,0,0,0.01])
        plt.xlim(fish.ul[0], fish.lr[0])
        plt.ylim(fish.ul[1], fish.lr[1])
        plt.subplot(2,2,3)
        plt.title('Area')
        plt.plot(area, 'm.', markersize=2, alpha=0.25)
        plt.ylim(0, 400)
        plt.subplot(2,2,4)
        plt.title('Heading')
        plt.plot(heading, 'g.', markersize=2, alpha=0.25)

        # Save
        plt.savefig(figure_path, dpi=180)
        print(i)
        plt.cla()
        plt.close()

#FIN
