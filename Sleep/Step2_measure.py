# -*- coding: utf-8 -*-
"""
Measure behaviour in a 96-well Sleep experiment

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

# Meaure behaviour for video paths (*.avi) in path_list
for path in path_list:
    # Create Paths
    video_path = base_path + path
    output_folder = os.path.dirname(video_path) + '/analysis'
    roi_path = output_folder + '/roi.csv'
    intensity_path = output_folder + '/intensity.csv'
    background_path = output_folder + '/background.png'

    # Create plate structure
    plate = MZF.create_plate()

    # Load ROIs
    plate = MZR.load_rois(roi_path, plate)

    # Load -Initial- Background Frame
    background = cv2.imread(background_path)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # Set backgrounds
    plate = MZF.set_backgrounds(background, plate)

    # Process behaviour
    intensity_roi = ((50,250), (80,850))
    plate, intensity = MZV.process_video_roi_analysis(video_path, plate, intensity_roi, 50000, output_folder)

    # Save fish behaviour
    fish_folder = output_folder + '/fish'
    if not os.path.exists(fish_folder):
        os.makedirs(fish_folder)
    for i, fish in enumerate(plate):
        fish_path = output_folder + f'/fish/{(i+1):02d}_fish.csv'
        fish_behaviour = np.zeros((len(fish.motion),4),dtype=np.float32)
        fish_behaviour[:,0] = np.array(fish.motion)
        fish_behaviour[:,1] = np.array(fish.x)
        fish_behaviour[:,2] = np.array(fish.y)
        fish_behaviour[:,3] = np.array(fish.area)
        np.savetxt(fish_path, fish_behaviour, delimiter=',')

    # Save intensity
    intensity_array = np.array(intensity)
    np.savetxt(intensity_path, intensity_array, fmt='%d')

#FIN
