# -*- coding: utf-8 -*-
"""
Analyse behaviour reponses in a 96-well PPI experiment

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
experiments = ['Akap11', 
               'Cacna1g', 
               #'Gria3', 
               'Grin2a',
               'Hcn4',
               'Herc1',
               'Nr3c2',
               'Sp4',
               'Trio',
               'Xpo7']

# Extract experiment behaviour
for experiment in experiments:
    plates, paths, controls, tests = MZU.parse_summary_PPI(summary_path, experiment)

    # Set list of video paths
    path_list = paths

    # Accumulators
    control_mean_single = []
    control_mean_paired = []
    test_mean_single = []
    test_mean_paired = []

    # Analyse behaviour for video paths (*.avi) in path_list
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
        figures_folder = output_folder + '/figures'
        controls_figure_folder = figures_folder + '/controls'
        tests_figure_folder = figures_folder + '/tests'
        experiment_analysis_folder = base_path + '/PPI' + '/analysis'

        # Empty figures folder
        MZU.clear_folder(figures_folder)
        MZU.create_folder(controls_figure_folder)
        MZU.create_folder(tests_figure_folder)
        
        # Create experiment analysis folder
        MZU.create_folder(experiment_analysis_folder)

        # Create plate structure
        name = path.split('/')[-1][:-4]
        plate = MZP.Plate(name)

        # Load stimulus (pulse) times
        pulses = np.load(responses_folder + '/pulses.npz')
        single_pulses = pulses['single_pulses']
        paired_pulses = pulses['paired_pulses']
        second_pulses = [x[1] for x in paired_pulses]

        # Default PPI response window
        pre_frames = 50
        post_frames = 100

        # Analyse control responses
        control_paths = glob.glob(controls_folder+'/*.npz')
        for path in control_paths:
            name = os.path.basename(path)[:-4]
            well_number = int(name.split('_')[1])
            behaviour = np.load(path)
            single_responses = behaviour['single_responses']
            paired_responses = behaviour['paired_responses']
            fish = plate.wells[well_number-1]

            # Responses to single pulse stimulus
            num_valid = 0
            num_responses = 0
            magnitudes = []
            distances = []
            turnings = []
            for i, pulse in enumerate(single_pulses):
                response = single_responses[:,:,i]
                is_valid = MZB.validate_response(response)
                if is_valid:
                    metrics = MZB.measure_response(response, 50)
                    is_response = MZB.classify_response(metrics)
                    num_valid += 1
                    magnitudes.append(metrics['magnitude'])
                    distances.append(metrics['distance'])
                    turnings.append(metrics['turning'])
                    if is_response:
                        num_responses += 1
            if num_valid < 2:
                invalid_single = True
            else:
                invalid_single = False
                response_probability = num_responses/num_valid
                mean_single = ([response_probability, np.nanmean(magnitudes), np.nanmean(distances), np.nanmean(np.abs(turnings))])

            # Responses to paired pulse stimulus
            num_valid = 0
            num_responses = 0
            magnitudes = []
            distances = []
            turnings = []
            for i, pair in enumerate(paired_pulses):
                response = paired_responses[:,:,i]
                pre_offset = pair[1] - pair[0]
                is_valid = MZB.validate_response(response)
                if is_valid:
                    first_metrics = MZB.measure_response(response, 50-pre_offset)
                    is_first_response = MZB.classify_response(first_metrics)
                    if is_first_response:
                        continue
                    num_valid += 1
                    second_metrics = MZB.measure_response(response, 50)
                    is_second_response = MZB.classify_response(second_metrics)
                    magnitudes.append(second_metrics['magnitude'])
                    distances.append(second_metrics['distance'])
                    turnings.append(second_metrics['turning'])
                    if is_second_response:
                        num_responses += 1
            if num_valid < 2:
                invalid_paired = True
            else:
                invalid_paired = False
                response_probability = num_responses/num_valid
                mean_paired = ([response_probability, np.nanmean(magnitudes), np.nanmean(distances), np.nanmean(np.abs(turnings))])

            # Accumulate controls
            if (not invalid_single) and (not invalid_paired):
                control_mean_single.append(mean_single)
                control_mean_paired.append(mean_paired)

        # Analyse test responses
        test_paths = glob.glob(tests_folder+'/*.npz')
        for path in test_paths:
            name = os.path.basename(path)[:-4]
            well_number = int(name.split('_')[1])
            behaviour = np.load(path)
            single_responses = behaviour['single_responses']
            paired_responses = behaviour['paired_responses']
            fish = plate.wells[well_number-1]

            # Responses to single pulse stimulus
            num_valid = 0
            num_responses = 0
            magnitudes = []
            distances = []
            turnings = []
            for i, pulse in enumerate(single_pulses):
                response = single_responses[:,:,i]
                is_valid = MZB.validate_response(response)
                if is_valid:
                    metrics = MZB.measure_response(response, 50)
                    is_response = MZB.classify_response(metrics)
                    num_valid += 1
                    magnitudes.append(metrics['magnitude'])
                    distances.append(metrics['distance'])
                    turnings.append(metrics['turning'])
                    if is_response:
                        num_responses += 1
            if num_valid < 2:
                invalid_single = True
            else:
                invalid_single = False
                response_probability = num_responses/num_valid
                mean_single = ([response_probability, np.nanmean(magnitudes), np.nanmean(distances), np.nanmean(np.abs(turnings))])

            # Responses to paired pulse stimulus
            num_valid = 0
            num_responses = 0
            magnitudes = []
            distances = []
            turnings = []
            for i, pair in enumerate(paired_pulses):
                response = paired_responses[:,:,i]
                pre_offset = pair[1] - pair[0]
                is_valid = MZB.validate_response(response)
                if is_valid:
                    first_metrics = MZB.measure_response(response, 50-pre_offset)
                    is_first_response = MZB.classify_response(first_metrics)
                    if is_first_response:
                        continue
                    num_valid += 1
                    second_metrics = MZB.measure_response(response, 50)
                    is_second_response = MZB.classify_response(second_metrics)
                    magnitudes.append(second_metrics['magnitude'])
                    distances.append(second_metrics['distance'])
                    turnings.append(second_metrics['turning'])
                    if is_second_response:
                        num_responses += 1
            if num_valid < 2:
                invalid_paired = True
            else:
                invalid_paired = False
                response_probability = num_responses/num_valid
                mean_paired = ([response_probability, np.nanmean(magnitudes), np.nanmean(distances), np.nanmean(np.abs(turnings))])

            # Accumulate tests
            if (not invalid_single) and (not invalid_paired):
                test_mean_single.append(mean_single)
                test_mean_paired.append(mean_paired)

    # Summarize Controls
    control_mean_single = np.array(control_mean_single)
    control_mean_paired = np.array(control_mean_paired)
    average_control_mean_single = np.nanmean(control_mean_single, axis=0)
    average_control_mean_paired = np.nanmean(control_mean_paired, axis=0)
    control_ppi = (average_control_mean_single[0] - average_control_mean_paired[0]) * 100.0

    # Summarize Controls
    test_mean_single = np.array(test_mean_single)
    test_mean_paired = np.array(test_mean_paired)
    average_test_mean_single = np.nanmean(test_mean_single, axis=0)
    average_test_mean_paired = np.nanmean(test_mean_paired, axis=0)
    test_ppi = (average_test_mean_single[0] - average_test_mean_paired[0]) * 100.0

    # Plot summary
    summary_figure_path = experiment_analysis_folder + f'/Summary_{experiment}.png'
    fig = plt.figure(figsize=(18, 10))
    plt.title(f'{experiment}: Control PPI ({control_ppi:.1f}%), Test PPI ({test_ppi:.1f}%)')
    plt.axis('off')
    for i in range(4):
        plt.subplot(2,4,i+1)
        plt.xlabel(f'Δ{(average_control_mean_single[i] - average_control_mean_paired[i]):.2f}')
        plt.plot([1,2], [control_mean_single[:,i], control_mean_paired[:,i]], color = [0.5,0.5,1.0,0.1])
        plt.boxplot([control_mean_single[:,i], control_mean_paired[:,i]], showmeans = True, meanline = True, notch=True)
    for i in range(4):
        plt.subplot(2,4,i+5)
        plt.xlabel(f'Δ{(average_test_mean_single[i] - average_test_mean_paired[i]):.2f}')
        num_fish = len(control_mean_single[:,i])
        plt.plot([1,2], [test_mean_single[:,i], test_mean_paired[:,i]], color = [0.5,0.5,1.0,0.1])
        plt.boxplot([test_mean_single[:,i], test_mean_paired[:,i]], showmeans = True, meanline = True, notch=True)
    plt.savefig(summary_figure_path, dpi=180)
    plt.close()

#FIN
