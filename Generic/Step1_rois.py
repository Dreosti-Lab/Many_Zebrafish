# -*- coding: utf-8 -*-
"""
Load user-defined ROIs for a generic zebrafish experiment

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

# Create Paths
video_path = '/home/kampff/data/Dropbox/Adam_Ele/schizo_fish/Raw_Videos/Non_Social.avi'
roi_path = video_path[:-4] + '_rois.csv'
output_folder = os.path.dirname(video_path) + '/analysis'

# Create plate
name = video_path.split('/')[-1][:-4]
plate = MZP.Plate(name, _rows=6, _cols=1)

# Load user-defined ROIs
plate.load_rois(roi_path)


#FIN
