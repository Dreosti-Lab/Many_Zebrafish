# -*- coding: utf-8 -*-
"""
Quickly check the result of a 96-well PPI experiment

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

# Import local modules
import MZ_video as MZV
import MZ_utilities as MZU

# Reload modules
import importlib
importlib.reload(MZV)
importlib.reload(MZU)
#----------------------------------------------------------

# Load list of video paths
path_list_path = base_path + "/PPI/path_list.txt"
path_list = MZU.load_path_list(path_list_path)

# Process summary images for video paths (*.avi) in path_list
for path in path_list:
    print(path)

    # Create Paths
    video_path = base_path + path
    output_folder = os.path.dirname(video_path) + '/analysis'

    # Create output folder
    MZU.create_folder(output_folder)

    # Generate initial background
    MZV.generate_initial_background(video_path, 200, -1, output_folder)   # 200 frames (throughout experiemnt)

    # Generate difference image
    MZV.generate_difference_image(video_path, 300, -1, output_folder)   # 300 frames (throughout experiemnt)

#FIN
