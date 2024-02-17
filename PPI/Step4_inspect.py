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
import MZ_bouts as MZB
import MZ_utilities as MZU

# Reload modules
import importlib
importlib.reload(MZP)
importlib.reload(MZV)
importlib.reload(MZB)
importlib.reload(MZU)
#----------------------------------------------------------

# Specify summary path
summary_path = "/home/kampff/data/Schizophrenia_data/Sumamry_Info.xlsx"

# Specify experiment abbreviation
experiment = 'Akap11'
#experiment = 'Cacna1g'
#experiment = 'Gria3'
#experiment = 'Grin2a'
#experiment = 'Hcn4'
#experiment = 'Herc1'
#experiment = 'Nr3c2'
#experiment = 'Sp4'
#experiment = 'Trio'
#experiment = 'Xpo7'
plates, paths, controls, tests = MZU.parse_summary_PPI(summary_path, experiment)

# Set list of video paths
path_list = paths

# Inspect behaviour for video paths (*.avi) in path_list
for p, path in enumerate(path_list):
    print(path)

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

    # Empty inspect folder
    MZU.clear_folder(inspect_folder)
    MZU.create_folder(controls_inspect_folder)
    MZU.create_folder(tests_inspect_folder)

    # Create plate structure
    name = path.split('/')[-1][:-4]
    plate = MZP.Plate(name)

    # Load ROIs
    plate.load_rois(roi_path)

    # Load stimulus (pulse) times
    pulses = np.load(responses_folder + '/pulses.npz')
    single_pulses = pulses['single_pulses']
    paired_pulses = pulses['paired_pulses']
    second_pulses = [x[1] for x in paired_pulses]

    # Default PPI responses
    pre_frames = 50
    post_frames = 100

    # Load video
    vid = cv2.VideoCapture(video_path)
    single_responses_frames = np.empty(8, object)
    paired_responses_frames = np.empty(8, object)
    
    # Load pre/post inspection frames from video
    for i, pulse in enumerate(single_pulses):
        frames = []
        ret = vid.set(cv2.CAP_PROP_POS_FRAMES, pulse-pre_frames)
        for f in range(pulse-pre_frames, pulse+post_frames):
            ret, frame = vid.read()
            frames.append(frame)
        single_responses_frames[i] = frames
    for i, pulse in enumerate(second_pulses):
        frames = []
        ret = vid.set(cv2.CAP_PROP_POS_FRAMES, pulse-pre_frames)
        for f in range(pulse-pre_frames, pulse+post_frames):
            ret, frame = vid.read()
            frames.append(frame)
        paired_responses_frames[i] = frames

    # Close video
    vid.release()

    # Feedback paremters
    clip_size = 128

    # Inspect control responses to single pulses
    control_paths = glob.glob(controls_folder+'/*.npz')
    for control_path in control_paths:
        name = os.path.basename(control_path)[:-4]
        well_number = int(name.split('_')[1])
        behaviour = np.load(control_path)
        single_responses = behaviour['single_responses']
        paired_responses = behaviour['paired_responses']
        fish = plate.wells[well_number-1]
        
        # Generate response video for each stimulus
        for i, pulse in enumerate(single_pulses):
            # Classify response
            response = single_responses[:,:,i]
            valid_bout = MZB.valid_bout(response)
            signal = response[4,:]

            # Prepare clip
            movie_path = controls_inspect_folder + f'/{name}_single_response_{i}.avi'
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            video = cv2.VideoWriter(movie_path, fourcc, 30, (clip_size,clip_size))

            # Render frames
            for frame_index in range(0, pre_frames+post_frames):
                frame = single_responses_frames[i][frame_index]
                crop = MZV.get_ROI_crop(frame, (fish.ul, fish.lr))
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (clip_size, clip_size))
                enhanced = cv2.normalize(resized, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

                x = response[0, frame_index]
                y = response[1, frame_index]
                heading = response[2, frame_index]
                motion = response[4, frame_index]
                offset = (fish.ul[0], fish.ul[1])
                scale = ((clip_size/crop.shape[1]), (clip_size/crop.shape[0]))

                rgb = MZB.plot_signal(rgb, signal, 2, 0.01, (255,0,255), 1, highlight=frame_index)
                rgb = MZB.draw_trajectory(rgb, response, offset, scale, 1, (0,255,0), 1)
                rgb = MZB.plot_fish(rgb, (x,y), heading, offset, scale, (255,0,0), 1)
                rgb = MZB.draw_response_type(rgb, valid_bout)
                ret = video.write(rgb)

            # Close clip
            ret = video.release()

            # Report
            print(f' - {name}: {i} - {valid_bout}')

#FIN
