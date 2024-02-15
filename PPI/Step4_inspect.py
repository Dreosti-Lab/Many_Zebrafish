# -*- coding: utf-8 -*-
"""
Inspect behaviour reponses in a 96-well PPI experiment

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
import glob
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

# Specify summary path
summary_path = "/home/kampff/data/Schizophrenia_data/Sumamry_Info.xlsx"

# Specify experiment abbreviation
experiment = 'Trio'
#experiment = 'Akap11'
plates, paths, controls, tests = MZU.parse_summary_PPI(summary_path, experiment)

# Set list of video paths
path_list = paths

# Inspect behaviour for video paths (*.avi) in path_list
for p, path in enumerate(path_list):
    # Create Paths
    video_path = base_path + path
    output_folder = os.path.dirname(video_path) + '/analysis'
    responses_folder = output_folder + '/responses'
    controls_folder = responses_folder + '/controls'
    tests_folder = responses_folder + '/tests'
    inspect_folder = output_folder + '/inspect'
    controls_inspect_folder = inspect_folder + '/controls'
    tests_inspect_folder = inspect_folder + '/tests'
    roi_path = output_folder + '/roi.csv'

    # Create inspect folder
    if not os.path.exists(inspect_folder):
        os.makedirs(inspect_folder)   

    # Create controls figure folder
    if not os.path.exists(controls_inspect_folder):
        os.makedirs(controls_inspect_folder)   

    # Create tests figure folder
    if not os.path.exists(tests_inspect_folder):
        os.makedirs(tests_inspect_folder)   

    # Create plate structure
    name = path.split('/')[-1][:-4]
    plate = MZP.Plate(name)

    # Load ROIs
    plate.load_rois(roi_path)

    # Load stimulus (pulse) times
    pulses = np.load(responses_folder + '/pulses.npz')
    single_pulses = pulses['single_pulses']
    paired_pulses = pulses['paired_pulses']

    # Default PPI responses
    pre_frames = 50
    post_frames = 150

    # Load video
    vid = cv2.VideoCapture(video_path)
    single_responses_frames = np.empty(8, object)
    paired_responses_frames = np.empty(8, object)
    
    # Load pre/post inspection frames from video
    for i, pulse in enumerate(single_pulses):
        frames = []
        vid.set(cv2.CAP_PROP_POS_FRAMES, pulse-pre_frames)
        for f in range(pulse-pre_frames, pulse+post_frames):
            ret, frame = vid.read()
            frames.append(frame)
        single_responses_frames[i] = frames
    for i, pair in enumerate(paired_pulses):
        pulse = pair[1]
        frames = []
        vid.set(cv2.CAP_PROP_POS_FRAMES, pulse-pre_frames)
        for f in range(pulse-pre_frames, pulse+post_frames):
            ret, frame = vid.read()
            frames.append(frame)
        paired_responses_frames[i] = frames

    # Close video
    vid.release()

    # Inspect control responses
    control_paths = glob.glob(controls_folder+'/*.npz')
    for control_path in control_paths:
        name = os.path.basename(control_path)[:-4]
        well_number = int(name.split('_')[1])
        behaviour = np.load(control_path)
        single_responses = behaviour['single_responses']
        paired_responses = behaviour['paired_responses']
        fish = plate.wells[well_number-1]
        for i, pulse in enumerate(single_pulses):
            movie_path = controls_inspect_folder + f'/{name}_single_response_{i}.avi'
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            video = cv2.VideoWriter(movie_path, fourcc, 30, (400,400))
            responses_drawn = []
            for frame_index in range(0, pre_frames+post_frames):
                frame = single_responses_frames[i][frame_index]
                crop = MZV.get_ROI_crop(frame, (fish.ul, fish.lr))
                resized = cv2.resize(crop, (400,400))
                x = int((single_responses[0, frame_index, i] - fish.ul[0]) * (400.0/crop.shape[1]))
                y = int((single_responses[1, frame_index, i] - fish.ul[1]) * (400.0/crop.shape[1]))
                motion = single_responses[4, frame_index, i]
                responses_drawn.append((frame_index*2, 395-int(motion/25)))
                for d in responses_drawn:
                    resized = cv2.circle(resized, d, 2, (0,0,255), 1)
                resized = cv2.circle(resized, (x,y), 3, (255,0,0), 1)
                ret = video.write(resized)
            ret = video.release()

#FIN
