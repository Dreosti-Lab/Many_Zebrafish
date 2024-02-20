# -*- coding: utf-8 -*-
"""
Quickly check the result of a generic zebrafish experiment

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

# Create Paths
video_path = '/home/kampff/data/Dropbox/Adam_Ele/schizo_fish/Raw_Videos/Non_Social.avi'
roi_path = video_path[:-4] + '_rois.csv'
output_folder = os.path.dirname(video_path) + '/analysis'

# Create output folder
MZU.create_folder(output_folder)

# Generate initial background
MZV.generate_initial_background(video_path, 200, -1, output_folder)   # 200 frames (throughout experiemnt)

# Generate difference image
MZV.generate_difference_image(video_path, 300, -1, output_folder)   # 300 frames (throughout experiemnt)

#FIN
