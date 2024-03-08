# -*- coding: utf-8 -*-
"""
Evaluate classification model for reponses in a 96-well PPI experiment

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
import MZ_classifier as MZC
import MZ_utilities as MZU

# Reload modules
import importlib
importlib.reload(MZP)
importlib.reload(MZV)
importlib.reload(MZB)
importlib.reload(MZC)
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

# Set model path
experiment_folder = base_path + '/PPI'
model_path = experiment_folder + '/classification_model.pt'

# Inspect behaviour for video paths (*.avi) in path_list
num_correct_algorithm = 0
num_correct_classifier = 0
num_total = 0
for p, path in enumerate(path_list):
    print(path)

    # Ignore bad paths (should fix in summary file!)
    if(path == '/gria3/231219/231219_grin2_PPI_Exp0.avi'): # Corrupt movie
        continue
    if(path == '/gria3/231219/231219_grin2_PPI_Exp0.avi'): # Corrupt movie
        continue
    if(path == '/grin2a/240219/exp0/240219_grin2_PPI_Exp000.avi'): # Not reviewed
        continue
    if(path == '/grin2a/240219/exp1/240219_grin2_PPI_Exp000.avi'): # Not reviewed
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
    controls_single_review_path = inspect_folder + '/controls_single_review.csv'
    controls_paired_review_path = inspect_folder + '/controls_paired_review.csv'
    tests_single_review_path = inspect_folder + '/tests_single_review.csv'
    tests_paired_review_path = inspect_folder + '/tests_paired_review.csv'
    roi_path = output_folder + '/roi.csv'

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

    # Open video
    vid = cv2.VideoCapture(video_path)

    # Create classifier
    classifier = MZC.Classifier(model_path)

    # Dataset parameters
    dataset_dim = 224
    dataset_times = [0, 9, 19]

    # Load response reviews
    controls_single_results = np.genfromtxt(controls_single_review_path, delimiter =",", dtype=str)
    controls_paired_results = np.genfromtxt(controls_paired_review_path, delimiter =",", dtype=str)
    tests_single_results = np.genfromtxt(tests_single_review_path, delimiter =",", dtype=str)
    tests_paired_results = np.genfromtxt(tests_paired_review_path, delimiter =",", dtype=str)

    # Process control single responses
    control_paths = controls_single_results[:,0]
    for i, control_path in enumerate(control_paths):
        name = control_path
        well_number = int(name.split('_')[1])
        plate_number = int(name.split('_')[3])
        response_number = int(name.split('_')[6])
        stimulus_frame = single_pulses[response_number]
        fish = plate.wells[well_number-1]
        algo_valid = controls_single_results[i, 1] == 'True'
        algo_response = controls_single_results[i, 2]  == 'True'
        is_valid = int(controls_single_results[i, 3])
        is_response = int(controls_single_results[i, 4])
        if not is_valid:
            continue
        prediction = classifier.classify(vid, (fish.roi_ul, fish.roi_lr), stimulus_frame)
        if(prediction == is_response):
            num_correct_classifier += 1
        if(algo_response == is_response):
            num_correct_algorithm += 1
        num_total += 1
        #print(('single', prediction, is_response, algo_response))

    # Process control paired responses
    control_paths = controls_paired_results[:,0]
    for i, control_path in enumerate(control_paths):
        name = control_path
        well_number = int(name.split('_')[1])
        plate_number = int(name.split('_')[3])
        response_number = int(name.split('_')[6])
        pulse = paired_pulses[response_number]
        first_stimulus_frame = pulse[0]
        second_stimulus_frame = pulse[1]
        fish = plate.wells[well_number-1]
        algo_valid = controls_single_results[i, 1] == 'True'
        algo_first_response = controls_single_results[i, 2]  == 'True'
        algo_second_response = controls_single_results[i, 3]  == 'True'
        is_valid = int(controls_paired_results[i, 4])
        is_first_response = int(controls_paired_results[i, 5])
        is_second_response = int(controls_paired_results[i, 6])
        if not is_valid:
            continue

        prediction = classifier.classify(vid, (fish.roi_ul, fish.roi_lr), first_stimulus_frame)
        if(prediction == is_first_response):
            num_correct_classifier += 1
        if(algo_first_response == is_first_response):
            num_correct_algorithm += 1
        num_total += 1
        #print(('first', prediction, is_first_response, algo_first_response))

        prediction = classifier.classify(vid, (fish.roi_ul, fish.roi_lr), second_stimulus_frame)
        if(prediction == is_second_response):
            num_correct_classifier += 1
        if(algo_second_response == is_second_response):
            num_correct_algorithm += 1
        num_total += 1
        #print(('second', prediction, is_second_response, algo_second_response))

    # Close video
    vid.release()

# Performance
algorithm_accuracy = (num_correct_algorithm/num_total) * 100.0
classifier_accuracy = (num_correct_classifier/num_total) * 100.0
print(f"Algorithm ({num_correct_algorithm} of {num_total}): {algorithm_accuracy}%")
print(f"Classifier ({num_correct_classifier} of {num_total}): {classifier_accuracy}%")

#FIN
