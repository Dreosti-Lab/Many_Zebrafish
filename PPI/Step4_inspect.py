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
summary_path = base_path + "/Sumamry_Info.xlsx"

# Specify experiment abbreviation
#experiment = 'Akap11'
#experiment = 'Cacna1g'
#experiment = 'Gria3'
experiment = 'Grin2a'
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

    # Ignore bad paths (should fix in summary file!)
    if(path == '/gria3/231219/231219_grin2_PPI_Exp0.avi'): # Corrupt movie
        continue
    if(path == '/nr3c2/231121/Exp0/231121_nr3c2_PPI_Exp0.avi'): # Bad LED (?)
        continue

    # Create Paths
    video_path = base_path + '/PPI' + path
    output_folder = os.path.dirname(video_path) + '/analysis'
    responses_folder = output_folder + '/responses'
    controls_folder = responses_folder + '/controls'
    tests_folder = responses_folder + '/tests'
    inspect_folder = output_folder + '/inspect'
    controls_inspect_folder = inspect_folder + '/controls'
    tests_inspect_folder = inspect_folder + '/tests'
    controls_single_review_path = inspect_folder + '/control_single_review.csv'
    controls_paired_review_path = inspect_folder + '/control_paired_review.csv'
    tests_single_review_path = inspect_folder + '/test_single_review.csv'
    tests_paired_review_path = inspect_folder + '/test_paired_review.csv'
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

    # Default PPI response window
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
        for f in range(pulse-pre_frames, pulse+post_frames+1):
            ret, frame = vid.read()
            frames.append(frame)
        single_responses_frames[i] = frames
    for i, pulse in enumerate(second_pulses):
        frames = []
        ret = vid.set(cv2.CAP_PROP_POS_FRAMES, pulse-pre_frames)
        for f in range(pulse-pre_frames, pulse+post_frames+1):
            ret, frame = vid.read()
            frames.append(frame)
        paired_responses_frames[i] = frames

    # Close video
    vid.release()

    # Inspect control responses
    control_paths = glob.glob(controls_folder+'/*.npz')
    single_results = []
    paired_results = []
    for control_path in control_paths:
        name = os.path.basename(control_path)[:-4]
        well_number = int(name.split('_')[1])
        behaviour = np.load(control_path)
        single_responses = behaviour['single_responses']
        paired_responses = behaviour['paired_responses']
        fish = plate.wells[well_number-1]
        
        # Generate response video for each single pulse stimulus
        for i, pulse in enumerate(single_pulses):
            response = single_responses[:,:,i]
            clip_path = controls_inspect_folder + f'/{name}_single_response_{i}.avi'
            result = MZB.inspect_response(single_responses_frames[i], (fish.roi_ul, fish.roi_lr), response, [50], clip_path)
            single_results.append(result)
            print(f' - {name}: {i} - sp')
        
        # Generate response video for each paired pulse stimulus
        for i, pair in enumerate(paired_pulses):
            first = pair[0]
            second = pair[1]
            response = paired_responses[:,:,i]
            clip_path = controls_inspect_folder + f'/{name}_paired_response_{i}.avi'
            result = MZB.inspect_response(paired_responses_frames[i], (fish.roi_ul, fish.roi_lr), response, [50-(second-first), 50], clip_path)
            paired_results.append(result)
            print(f' - {name}: {i} - pp')
    
    # Save results (for review)
    np.savetxt(controls_single_review_path, single_results, delimiter =",", fmt ='%s')
    np.savetxt(controls_paired_review_path, paired_results, delimiter =",", fmt ='%s')

    # Inspect test responses
    test_paths = glob.glob(tests_folder+'/*.npz')
    single_results = []
    paired_results = []
    for test_path in test_paths:
        name = os.path.basename(test_path)[:-4]
        well_number = int(name.split('_')[1])
        behaviour = np.load(test_path)
        single_responses = behaviour['single_responses']
        paired_responses = behaviour['paired_responses']
        fish = plate.wells[well_number-1]
        
        # Generate response video for each single pulse stimulus
        for i, pulse in enumerate(single_pulses):
            response = single_responses[:,:,i]
            clip_path = tests_inspect_folder + f'/{name}_single_response_{i}.avi'
            result = MZB.inspect_response(single_responses_frames[i], (fish.roi_ul, fish.roi_lr), response, [50], clip_path)
            single_results.append(result)
            print(f' - {name}: {i} - sp')

        # Generate response video for each paired pulse stimulus
        for i, pair in enumerate(paired_pulses):
            first = pair[0]
            second = pair[1]
            response = paired_responses[:,:,i]
            clip_path = tests_inspect_folder + f'/{name}_paired_response_{i}.avi'
            result = MZB.inspect_response(paired_responses_frames[i], (fish.roi_ul, fish.roi_lr), response, [50-(second-first), 50], clip_path)
            paired_results.append(result)
            print(f' - {name}: {i} - pp')
    
    # Save results (for review)
    np.savetxt(tests_single_review_path, single_results, delimiter =",", fmt ='%s')
    np.savetxt(tests_paired_review_path, paired_results, delimiter =",", fmt ='%s')

#FIN
