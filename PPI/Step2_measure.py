 # -*- coding: utf-8 -*-
"""
Measure behaviour in a 96-well PPI experiment

@author: kampff
"""
#----------------------------------------------------------
# Load environment file and variables
import os
from dotenv import load_dotenv
load_dotenv()
libs_path = os.getenv('LIBS_PATH')
base_path = os.getenv('BASE_PATH')

# Set library paths
import sys
sys.path.append(libs_path)

# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Import modules
import MZ_plate as MZP
import MZ_video as MZV
import MZ_utilities as MZU

# Reload modules
import importlib
importlib.reload(MZP)
importlib.reload(MZV)
importlib.reload(MZU)
#----------------------------------------------------------

# Load list of video paths
path_list_path = base_path + "/path_list.txt"
path_list = MZU.load_path_list(path_list_path)

# Meaure behaviour for video paths (*.avi) in path_list
for path in path_list:
    # Create Paths
    video_path = base_path + path
    output_folder = os.path.dirname(video_path) + '/analysis'
    validation_folder = output_folder + '/validation'
    roi_path = output_folder + '/roi.csv'
    background_path = output_folder + '/background.png'

    # Create plate
    name = path.split('/')[-1][:-4]
    plate = MZP.Plate(name)

    # Load ROIs
    plate.load_rois(roi_path)

    # Load -Initial- Background Frame
    background = cv2.imread(background_path)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # Initialize backgrounds
    plate.init_backgrounds(background)

    # Process behaviour
    led_roi = ((0,0), (48,48))
    plate = MZV.fish_tracking_roi(video_path, plate, led_roi, start_frame=0, end_frame=-1, max_background_rate=400, validate=True, validation_folder=validation_folder)

    # Save plate
    plate.save(output_folder)
    
#FIN
