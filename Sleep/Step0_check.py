# -*- coding: utf-8 -*-
"""
Quickly check the result of a 96-well Sleep experiment

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

# Import local modules
import MZ_video as MZV
import MZ_utilities as MZU

# Reload modules
import importlib
importlib.reload(MZV)
importlib.reload(MZU)

# Load list of video paths
path_list_path = base_path + "/Sleep_Behaviour/path_list.txt"
path_list = MZU.load_path_list(path_list_path)

# Process summary images for video paths (*.avi) in path_list
for path in path_list:
    # Create Paths
    video_path = base_path + path
    output_folder = os.path.dirname(video_path) + '/analysis'

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process
    MZV.process_video_summary_images(video_path, 300, 1000, output_folder)

#FIN
