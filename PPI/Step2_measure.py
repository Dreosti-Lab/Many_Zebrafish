# -*- coding: utf-8 -*-
"""
Measure behaviour in a 96-well PPI experiment

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

# Meaure behaviour for video paths (*.avi) in path_list
for path in path_list:
    # Create Paths
    video_path = base_path + path
    output_folder = os.path.dirname(video_path) + '/analysis'
    roi_path = output_folder + '/roi.csv'
    led_path = output_folder + '/led.csv'
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
    led_roi = ((0,0), (48,48))
    plate, led_intensity = MZV.fish_tracking_roi(video_path, plate, led_roi, -1, 400, True, output_folder)

    # Save plate
    MZF.save_plate(plate, led_intensity, output_folder)

#FIN
