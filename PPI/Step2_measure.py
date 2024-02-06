# -*- coding: utf-8 -*-
"""
Measure behaviour in a 96-well experiment

@author: kamnpff (Adam Kampff)
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'/home/kampff/Repos/Dreosti-Lab/Many_Zebrafish/libs'
#-----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Set "Base Path" for this analysis session
base_path = r'/run/media/kampff/Data/Zebrafish/'
#base_path = r'\\128.40.155.187\data\D R E O S T I   L A B'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
sys.path.append(lib_path)

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

# ----Load Folder List Here----

# Input path
input_path = base_path + r'/PPI_Behaviour/240123_nr3c2_PPI_Exp000.avi'
#input_path = base_path + r'/PPI_Behaviour/231218_herc1_PPI_stim_Exp00.avi'

# Output folder
output_folder = base_path + r'/PPI_Behaviour/output'

# Create plate structure
plate = MZF.create_plate()

# Load ROIs
roi_path = output_folder + '/roi.csv'
plate = MZR.load_rois(roi_path, plate)

# Load -Initial- Background Frame
background_path = output_folder + r'/background.png'
background = cv2.imread(background_path)
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# Set backgrounds
plate = MZF.set_backgrounds(background, plate)

# Process behaviour
plate, led_intensity = MZV.process_video_roi_analysis(input_path, plate, output_folder)

# Extract stimulus times and types
single_pulses, paired_pulses = MZU.extract_ppi_stimuli(led_intensity)

# Plot LED
plt.plot(np.array(led_intensity))
plt.show()

# Save fish motion
fish_folder = output_folder + '/fish'
if not os.path.exists(fish_folder):
    os.makedir(fish_folder)
for i, fish in enumerate(plate):
    fish_path = output_folder + f'/fish/{(i+1):02d}_fish.txt'
    motion_array = np.array(fish.motion)
    np.savetxt(fish_path, motion_array, fmt='%d')

# Analyse responses
pre_frames = 100
post_frames = 300
num_frames = len(led_intensity)
num_fish = 96
motion = np.zeros((num_fish, num_frames))
for i, fish in enumerate(plate):
    motion[i,:] = fish.motion

single_responses = np.zeros((len(single_pulses), pre_frames+post_frames))
for i, p in enumerate(single_pulses):
    single_responses[i, :] = np.mean(motion[:, (p-(pre_frames+1)):(p+post_frames-1)], axis=0)
plt.subplot(1,2,1)
plt.vlines(pre_frames, 0, np.max(single_responses), color=[0,1,0,1])
plt.plot(single_responses.T, color=[0,0,0,0.25])
plt.plot(np.mean(single_responses, axis=0), color=[1.0,0,0,1.0])

paired_responses = np.zeros((len(paired_pulses), pre_frames+post_frames))
for i, pair in enumerate(paired_pulses):
    p = pair[0]
    paired_responses[i, :] = np.mean(motion[:, (p-(pre_frames+1)):(p+post_frames-1)], axis=0)
plt.subplot(1,2,2)
plt.vlines(pre_frames, 0, np.max(paired_responses), color=[0,1,0,1])
plt.vlines(pre_frames+30, 0, np.max(paired_responses), color=[0,1,0,1])
plt.plot(paired_responses.T, color=[0,0,0,0.25])
plt.plot(np.mean(paired_responses, axis=0), color=[1.0,0,0,1.0])
plt.show()

#FIN
