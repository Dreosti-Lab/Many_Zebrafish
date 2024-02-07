# -*- coding: utf-8 -*-
"""
Automatically extract ROIs from a 96-well experiment

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
import MZ_fish as MZF
import MZ_roi as MZR
import MZ_utilities as MZU

# Reload modules
import importlib
importlib.reload(MZF)
importlib.reload(MZR)
importlib.reload(MZU)

# Load list of video paths
path_list_path = base_path + "/PPI_Behaviour/path_list.txt"
path_list = MZU.load_path_list(path_list_path)

# Automatically detect ROIs for video paths (*.avi) in path_list
for path in path_list:
    # Create Paths
    output_folder = os.path.dirname(base_path + path) + '/analysis'
    image_path = output_folder + '/difference.png'

    # Create plate structure
    plate = MZF.create_plate()

    # Automatically detect ROIs
    plate = MZR.find_rois(image_path, plate, output_folder)

#FIN
