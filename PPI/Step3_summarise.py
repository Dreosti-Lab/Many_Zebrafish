# -*- coding: utf-8 -*-
"""
Summarise rseponses in a 96-well PPI experiment

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
import MZ_video as MZV
import MZ_classifier as MZC
import MZ_utilities as MZU

# Reload modules
import importlib
importlib.reload(MZP)
importlib.reload(MZV)
importlib.reload(MZC)
importlib.reload(MZU)
#----------------------------------------------------------

# Specify summary path
summary_path = base_path + "/Sumamry_Info.xlsx"

# Specify experiment abbreviation
experiments = ['Akap11', 
               'Cacna1g', 
               #'Gria3', 
               'Grin2a',
               'Hcn4',
               'Herc1',
               'Nr3c2',
               'Sp4',
               'Trio',
               'Xpo7'
               ]

# Default PPI response window
pre_frames = 50
post_frames = 100

# Default dataset dimensions and times
data_dim = 224
data_times = [0, 9, 19]

# Extract experiment
for experiment in experiments:
    plates, paths, controls, tests = MZU.parse_summary_PPI(summary_path, experiment)

    # Set list of video paths
    path_list = paths

    # Extract behaviour for video paths (*.avi) in path_list
    control_single_responses = np.empty((8,pre_frames+post_frames+1,0))
    test_single_responses = np.empty((8,pre_frames+post_frames+1,0))
    control_paired_responses = np.empty((8,pre_frames+post_frames+1,0))
    test_paired_responses = np.empty((8,pre_frames+post_frames+1,0))
    for p, path in enumerate(path_list):
        print(path)

        # Ignore bad paths (should fix in summary file!)
        if(path == '/gria3/231219/231219_grin2_PPI_Exp0.avi'): # Corrupt movie
            continue
        if(path == '/gria3/240213/exp 1/240213_gria3_PPI_Exp00.avi'): # Bad LED (?)
            continue
        if(path == '/gria3/240213/expo 0/240213_gria3_PPI_Exp00.avi'): # Bad LED (?)
            continue
        if(path == '/nr3c2/231121/Exp0/231121_nr3c2_PPI_Exp0.avi'): # Bad LED (?)
            continue
        if(path == '/sp4/231116/Exp0/231116_SP4_Exp00.avi'): # Bad LED (?)
            continue

        # Specify Paths
        video_path = base_path + '/PPI' + path
        output_folder = os.path.dirname(video_path) + '/analysis'
        responses_folder = output_folder + '/responses'
        controls_folder = responses_folder + '/controls'
        tests_folder = responses_folder + '/tests'

        # Empty responses folder
        MZU.clear_folder(responses_folder)
        MZU.create_folder(controls_folder)
        MZU.create_folder(tests_folder)

        # Create plate
        name = path.split('/')[-1][:-4]
        plate = MZP.Plate(name)

        # Load plate
        print(f"Loading plate data...{p+1} of {len(path_list)}")
        plate.load(output_folder)

        # Extract stimulus times and types
        print("Extracting pulses...")
        led_intensity = plate.intensity
        single_pulses, paired_pulses = MZU.extract_ppi_stimuli(led_intensity)
        first_pulses = [x[0] for x in paired_pulses]
        second_pulses = [x[1] for x in paired_pulses]
        np.savez(responses_folder + '/pulses.npz', single_pulses=single_pulses, paired_pulses=paired_pulses)

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
        plt.savefig(output_folder + f'/led_intensity.png', dpi=180)
        plt.cla()
        plt.close()

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
        
        # Extract and save all control responses
        for c in controls[p]:
            fish = plate.wells[c-1]
            behaviour = MZU.extract_behaviour(plate, c-1)
            single_responses = MZU.extract_responses(behaviour, single_pulses, pre_frames, post_frames)
            paired_responses = MZU.extract_responses(behaviour, second_pulses, pre_frames, post_frames)
            single_datapoints = []
            paired_datapoints = []
            for r in range(8):
                single_datapoint = MZC.generate_datapoint(single_responses_frames[r], (fish.roi_ul, fish.roi_lr), data_dim, data_times, pre_frames)
                first_frame_offset = pre_frames - (second_pulses[r] - first_pulses[r])
                first_datapoint = MZC.generate_datapoint(paired_responses_frames[r], (fish.roi_ul, fish.roi_lr), data_dim, data_times, first_frame_offset)
                second_datapoint = MZC.generate_datapoint(paired_responses_frames[r], (fish.roi_ul, fish.roi_lr), data_dim, data_times, pre_frames)
                paired_datapoint = []
                paired_datapoint.append(first_datapoint)
                paired_datapoint.append(second_datapoint)
                single_datapoints.append(single_datapoint)
                paired_datapoints.append(paired_datapoint)
            control_path = controls_folder + f'/control_{c}_plate_{plates[p]}.npz'
            np.savez(control_path, single_responses=single_responses, paired_responses=paired_responses, single_datapoints=single_datapoints, paired_datapoints=paired_datapoints)

        # Extract and save all test responses
        for t in tests[p]:
            fish = plate.wells[t-1]
            behaviour = MZU.extract_behaviour(plate, t-1)
            single_responses = MZU.extract_responses(behaviour, single_pulses, pre_frames, post_frames)
            paired_responses = MZU.extract_responses(behaviour, second_pulses, pre_frames, post_frames)
            single_datapoints = []
            paired_datapoints = []
            for r in range(8):
                single_datapoint = MZC.generate_datapoint(single_responses_frames[r], (fish.roi_ul, fish.roi_lr), data_dim, data_times, pre_frames)
                first_frame_offset = pre_frames - (second_pulses[r] - first_pulses[r])
                first_datapoint = MZC.generate_datapoint(paired_responses_frames[r], (fish.roi_ul, fish.roi_lr), data_dim, data_times, first_frame_offset)
                second_datapoint = MZC.generate_datapoint(paired_responses_frames[r], (fish.roi_ul, fish.roi_lr), data_dim, data_times, pre_frames)
                paired_datapoint = []
                paired_datapoint.append(first_datapoint)
                paired_datapoint.append(second_datapoint)
                single_datapoints.append(single_datapoint)
                paired_datapoints.append(paired_datapoint)
            test_path = tests_folder + f'/test_{t}_plate_{plates[p]}.npz'
            np.savez(test_path, single_responses=single_responses, paired_responses=paired_responses, single_datapoints=single_datapoints, paired_datapoints=paired_datapoints)

#FIN
