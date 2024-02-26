 # -*- coding: utf-8 -*-
"""
Visualise behaviour in a generic zebrafish experiment

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
import MZ_fish as MZF
import MZ_video as MZV
import MZ_bouts as MZB
import MZ_utilities as MZU

# Reload modules
import importlib
importlib.reload(MZP)
importlib.reload(MZF)
importlib.reload(MZV)
importlib.reload(MZB)
importlib.reload(MZU)
#----------------------------------------------------------

# Create Paths
video_path = '/home/kampff/data/Dropbox/Adam_Ele/schizo_fish/Raw_Videos/Non_Social.avi'
output_folder = os.path.dirname(video_path) + '/analysis'
visualise_path = output_folder + '/visualise.avi'
roi_path = video_path[:-4] + '_rois.csv'

# Create plate
name = video_path.split('/')[-1][:-4]
plate = MZP.Plate(name, _rows=6, _cols=1)

# Load user-defined ROIs
plate.load_rois(roi_path)
plate.save_rois(output_folder + '/roi.csv')

# Load behaviour data
plate.load(output_folder)

# Extract all behaviour
behaviours = []
for i in range(plate.num_wells):
    behaviour = MZU.extract_behaviour(plate, i)
    behaviours.append(behaviour)

# Open input video
vid = cv2.VideoCapture(video_path)
num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set start and end frame
start_frame = 0
num_frames = 600

# Crate visualise movie
fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
output = cv2.VideoWriter(visualise_path, fourcc, 30, (frame_width,frame_height))

# Visualise
f = start_frame
accumulator = np.zeros((frame_height, frame_width))
for i in range(num_frames):
    ret = vid.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, im = vid.read()
    for j, fish in enumerate(plate.wells):
        # Draw current position
        im = MZB.draw_fish(im, (fish.x[f], fish.y[f]), fish.heading[f], (0,0), (1,1), (0,255,255), 1)
        # Draw history
        if i > 100:
            im = MZB.draw_response_trajectory(im, behaviours[j][100:f,:], (0,0), (1,1), 1, (0,0,255),1)
    ret = output.write(im)
    if i > 100:
        f = f + ((i - 90)  // 10)
    else:
        f = f + 1
    print(i, f)
ret = vid.release()
ret = output.release()
#FIN
