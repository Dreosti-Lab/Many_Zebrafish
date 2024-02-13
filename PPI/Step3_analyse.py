# -*- coding: utf-8 -*-
"""
Analyse behaviour in a 96-well PPI experiment

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

# Specify summary path
summary_path = "/home/kampff/data/Schizophrenia_data/Sumamry_Info.xlsx"

# Specify experiment abbreviation
experiment = 'Herc1'
experiment = 'Akap11'
plates, paths, controls, tests = MZU.parse_summary_PPI(summary_path, experiment)

# Set list of video paths
path_list = paths
path = path_list[0]

# Analyse behaviour for video paths (*.avi) in path_list
control_single_responses = np.empty((8,200,0))
test_single_responses = np.empty((8,200,0))
control_paired_responses = np.empty((8,200,0))
test_paired_responses = np.empty((8,200,0))
for p, path in enumerate(path_list):
    # Create Paths
    video_path = base_path + path
    output_folder = os.path.dirname(video_path) + '/analysis'
    figures_folder = os.path.dirname(video_path) + '/analysis/figures'
    fish_figures_folder = os.path.dirname(video_path) + '/analysis/figures/fish'
    roi_path = output_folder + '/roi.csv'
    led_path = output_folder + '/intensity.csv'
    background_path = output_folder + r'/background.png'

    # Create figures folder
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)   

    # Create fish figures folder
    if not os.path.exists(fish_figures_folder):
        os.makedirs(fish_figures_folder)   

    # Create plate
    name = path.split('/')[-1][:-4]
    plate = MZP.Plate(name)

    # Load plate
    print(f"Loading plate data...{p} of {len(path_list)}")
    plate.load(output_folder)

    # Extract stimulus times and types
    led_intensity = plate.intensity
    print("Extracting pulses...")
    single_pulses, paired_pulses = MZU.extract_ppi_stimuli(led_intensity)

    # Plot LED
    print("Plotting LED...")
    fig = plt.figure(figsize=(10, 8))
    plt.title('LED-based Pulse Detection')
    for i,pulse in enumerate(single_pulses):
        plt.subplot(2,8,i+1)
        plt.plot(led_intensity[(pulse-100):(pulse+100)])
        plt.plot(100, np.max(led_intensity), 'r+')
        plt.yticks([])
    for i,pair in enumerate(paired_pulses):
        plt.subplot(2,8,i+9)
        plt.plot(led_intensity[(pair[0]-100):(pair[0]+100)])
        plt.plot(100, np.max(led_intensity), 'g+')
        plt.plot((pair[1]-pair[0])+100, np.max(led_intensity), 'b+')
        plt.yticks([])
    plt.savefig(figures_folder + '/led_intensity.png', dpi=180)
    plt.cla()
    plt.close()

    # PPI responses
    pre_frames = 50
    post_frames = 150
    single_responses = np.zeros((len(single_pulses), pre_frames+post_frames))
    paired_responses = np.zeros((len(paired_pulses), pre_frames+post_frames))

    # Extract all control responses
    for c in controls[p]:
        motion = plate.wells[c-1].motion
        for j, pulse in enumerate(single_pulses):
            single_responses[j, :] = motion[(pulse-(pre_frames+1)):(pulse+post_frames-1)]
        for j, pair in enumerate(paired_pulses):
            pulse = pair[0]
            paired_responses[j, :] = motion[(pulse-(pre_frames+1)):(pulse+post_frames-1)]
        control_single_responses = np.dstack((control_single_responses, single_responses))
        control_paired_responses = np.dstack((control_paired_responses, paired_responses))

    # Extract all test responses
    for t in tests[p]:
        motion = plate.wells[t-1].motion
        for j, pulse in enumerate(single_pulses):
            single_responses[j, :] = motion[(pulse-(pre_frames+1)):(pulse+post_frames-1)]
        for j, pair in enumerate(paired_pulses):
            pulse = pair[0]
            paired_responses[j, :] = motion[(pulse-(pre_frames+1)):(pulse+post_frames-1)]
        test_single_responses = np.dstack((test_single_responses, single_responses))
        test_paired_responses = np.dstack((test_paired_responses, paired_responses))

# Summarize
control_single_averages = np.mean(control_single_responses, axis=0)
control_paired_averages = np.mean(control_paired_responses, axis=0)
test_single_averages = np.mean(test_single_responses, axis=0)
test_paired_averages = np.mean(test_paired_responses, axis=0)

control_single_mean = np.mean(control_single_averages, axis=1)
control_paired_mean = np.mean(control_paired_averages, axis=1)
test_single_mean = np.mean(test_single_averages, axis=1)
test_paired_mean = np.mean(test_paired_averages, axis=1)

fig = plt.figure(figsize=(10, 8))
plt.subplot(2,2,1)
plt.plot(control_single_averages, color = [0, 0, 0, 0.1])
plt.plot(control_single_mean, linewidth=2, color = [1, 0, 0, 0.75])
plt.subplot(2,2,2)
plt.plot(control_paired_averages, color = [0, 0, 0, 0.1])
plt.plot(control_paired_mean, linewidth=2, color = [1, 0, 0, 0.75])
plt.subplot(2,2,3)
plt.plot(test_single_averages, color = [0, 0, 0, 0.1])
plt.plot(test_single_mean, linewidth=2, color = [1, 0, 0, 0.75])
plt.subplot(2,2,4)
plt.plot(test_paired_averages, color = [0, 0, 0, 0.1])
plt.plot(test_paired_mean, linewidth=2, color = [1, 0, 0, 0.75])
plt.show()

#    # Analyse
#    for i, fish in enumerate(plate.wells):
#        figure_path = fish_figures_folder + f'/{(i+1):02d}_fish.png'
#        x = fish.x
#        y = fish.y
#        area = fish.area
#        heading = fish.heading
#        motion = fish.motion
#
#        fig = plt.figure(figsize=(10, 8))
#        plt.subplot(2,3,1)
#        plt.title('Motion')
#        plt.plot(motion)
#        plt.subplot(2,3,2)
#        plt.title('Tracking')
#        plt.plot(x,y,'.', markersize=3, color=[0,0,0,0.01])
#        plt.xlim(fish.ul[0], fish.lr[0])
#        plt.ylim(fish.ul[1], fish.lr[1])
#        plt.subplot(2,3,3)
#        plt.title('Area')
#        plt.plot(area, 'm.', markersize=2, alpha=0.25)
#        plt.subplot(2,3,6)
#        plt.title('Heading')
#        plt.plot(heading, 'g.', markersize=2, alpha=0.25)
#
#        # PPI responses
#        pre_frames = 50
#        post_frames = 150
#        single_responses = np.zeros((len(single_pulses), pre_frames+post_frames))
#        for j, p in enumerate(single_pulses):
#            single_responses[j, :] = motion[(p-(pre_frames+1)):(p+post_frames-1)]
#        plt.subplot(2,3,4)
#        plt.title('Single Pulse')
#        plt.vlines(pre_frames, 0, np.max(single_responses), color=[0,1,0,1])
#        plt.plot(single_responses.T, color=[0,0,0,0.25])
#        plt.plot(np.mean(single_responses, axis=0), color=[1.0,0,0,1.0])
#        plt.ylim(0, np.max(motion))
#        paired_responses = np.zeros((len(paired_pulses), pre_frames+post_frames))
#        for j, pair in enumerate(paired_pulses):
#            p = pair[0]
#            paired_responses[j, :] = motion[(p-(pre_frames+1)):(p+post_frames-1)]
#        plt.subplot(2,3,5)
#        plt.title('Paired Pulse')
#        plt.vlines(pre_frames, 0, np.max(paired_responses), color=[0,1,0,1])
#        plt.vlines(pre_frames+30, 0, np.max(paired_responses), color=[0,1,0,1])
#        plt.plot(paired_responses.T, color=[0,0,0,0.25])
#        plt.plot(np.mean(paired_responses, axis=0), color=[1.0,0,0,1.0])
#        plt.ylim(0, np.max(motion))
#
#        # Save
#        plt.savefig(figure_path, dpi=180)
#        print(f'Analysing Fish {i} of 96')
#        plt.cla()
#        plt.close()

#FIN
