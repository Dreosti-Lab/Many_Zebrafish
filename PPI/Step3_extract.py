# -*- coding: utf-8 -*-
"""
Extract rseponses in a 96-well PPI experiment

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

# Set model path
experiment_folder = base_path + '/PPI'
model_path = experiment_folder + '/classification_model.pt'

# Create classifier
classifier = MZC.Classifier(model_path)

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

# Extract experiment behaviour
for experiment in experiments:
    plates, paths, controls, tests = MZU.parse_summary_PPI(summary_path, experiment)

    # Set list of video paths
    path_list = paths

    # Analyse behaviour for video paths (*.avi) in path_list
    control_single_responses = np.empty((8,200,0))
    test_single_responses = np.empty((8,200,0))
    control_paired_responses = np.empty((8,200,0))
    test_paired_responses = np.empty((8,200,0))
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
        plt.savefig(output_folder + f'/led_intensity_plate_{plates[p]}.png', dpi=180)
        plt.cla()
        plt.close()

        # Extract PPI responses
        pre_frames = 50
        post_frames = 100
        
        # Open Video
        vid = cv2.VideoCapture(video_path)
        
        # Extract and save all control responses
        for c in controls[p]:
            behaviour = MZU.extract_behaviour(plate, c-1)
            single_responses = MZU.extract_responses(behaviour, single_pulses, pre_frames, post_frames)
            paired_responses = MZU.extract_responses(behaviour, second_pulses, pre_frames, post_frames)
            fish = plate.wells[c-1]
            single_classifications = []
            first_classifications = []
            second_classifications = []
            for r in range(8):
                single_classifications.append(classifier.classify(vid, (fish.roi_ul, fish.roi_lr), single_pulses[r]))
                first_classifications.append(classifier.classify(vid, (fish.roi_ul, fish.roi_lr), first_pulses[r]))
                second_classifications.append(classifier.classify(vid, (fish.roi_ul, fish.roi_lr), second_pulses[r]))
            control_path = controls_folder + f'/control_{c}_plate_{plates[p]}.npz'
            np.savez(control_path, single_responses=single_responses, paired_responses=paired_responses, single_classifications=single_classifications, first_classifications=first_classifications, second_classifications=second_classifications)

        # Extract and save all test responses
        for t in tests[p]:
            behaviour = MZU.extract_behaviour(plate, t-1)
            single_responses = MZU.extract_responses(behaviour, single_pulses, pre_frames, post_frames)
            paired_responses = MZU.extract_responses(behaviour, second_pulses, pre_frames, post_frames)
            fish = plate.wells[t-1]
            single_classifications = []
            first_classifications = []
            second_classifications = []
            for r in range(8):
                single_classifications.append(classifier.classify(vid, (fish.roi_ul, fish.roi_lr), single_pulses[r]))
                first_classifications.append(classifier.classify(vid, (fish.roi_ul, fish.roi_lr), first_pulses[r]))
                second_classifications.append(classifier.classify(vid, (fish.roi_ul, fish.roi_lr), second_pulses[r]))
            test_path = tests_folder + f'/test_{t}_plate_{plates[p]}.npz'
            np.savez(test_path, single_responses=single_responses, paired_responses=paired_responses, single_classifications=single_classifications, first_classifications=first_classifications, second_classifications=second_classifications)

        # Close video
        vid.release()
#FIN
