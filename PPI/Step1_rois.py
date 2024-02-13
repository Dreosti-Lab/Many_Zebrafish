# -*- coding: utf-8 -*-
"""
Automatically extract ROIs from a 96-well PPI experiment

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
import MZ_utilities as MZU

# Reload modules
import importlib
importlib.reload(MZP)
importlib.reload(MZU)
#----------------------------------------------------------

# Load list of video paths
path_list_path = base_path + "/path_list.txt"
path_list = MZU.load_path_list(path_list_path)

# Automatically detect ROIs for video paths (*.avi) in path_list
for path in path_list:
    print(path)

    # Create Paths
    output_folder = os.path.dirname(base_path + path) + '/analysis'
    image_path = output_folder + '/difference.png'
    roi_path = output_folder + '/roi.csv'

    # Create plate
    name = path.split('/')[-1][:-4]
    plate = MZP.Plate(name)

    # Automatically detect ROIs
    plate.find_rois(image_path, 5, output_folder)

    # Store ROIs
    plate.save_rois(roi_path)

#FIN
