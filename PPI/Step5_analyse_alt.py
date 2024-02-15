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
#experiment = 'Akap11'
experiment = 'Cacna1g'
#experiment = 'Gria3'
#experiment = 'Grin2a'
#experiment = 'Hcn4'
#experiment = 'Herc1'
#experiment = 'Nr3c2'
#experiment = 'Sp4'
#experiment = 'Trio'
#experiment = 'Xpo7'
#experiment_folder = base_path + '/akap11'
experiment_folder = base_path + '/cacna1g'
#experiment_folder = base_path + '/gria3'
#experiment_folder = base_path + '/grin2a'
#experiment_folder = base_path + '/hcn4'
#experiment_folder = base_path + '/herc1'
#experiment_folder = base_path + '/nr3c2'
#experiment_folder = base_path + '/sp4'
#experiment_folder = base_path + '/trio'
#experiment_folder = base_path + '/xpo7'

plates, paths, controls, tests = MZU.parse_summary_PPI(summary_path, experiment)

# Set list of video paths
path_list = paths
#path_list.remove(path_list[0])

# Accumulators
control_mean_single = []
control_mean_paired = []
test_mean_single = []
test_mean_paired = []

# Analyse behaviour for video paths (*.avi) in path_list
for p, path in enumerate(path_list):
    # Create Paths
    video_path = base_path + path
    output_folder = os.path.dirname(video_path) + '/analysis'
    responses_folder = output_folder + '/responses'
    controls_folder = responses_folder + '/controls'
    tests_folder = responses_folder + '/tests'
    figures_folder = output_folder + '/figures'
    controls_figure_folder = figures_folder + '/controls'
    tests_figure_folder = figures_folder + '/tests'
    experiment_analysis_folder = experiment_folder + '/analysis'

    # Create figures folder
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)   

    # Create controls figure folder
    if not os.path.exists(controls_figure_folder):
        os.makedirs(controls_figure_folder)   

    # Create tests figure folder
    if not os.path.exists(tests_figure_folder):
        os.makedirs(tests_figure_folder)   

    # Create experiment analysis folder
    if not os.path.exists(experiment_analysis_folder):
        os.makedirs(experiment_analysis_folder)   

    # Load stimulus (pulse) times
    pulses = np.load(responses_folder + '/pulses.npz')
    single_pulses = pulses['single_pulses']
    paired_pulses = pulses['paired_pulses']

    # Plot control responses (only if they don't respond to first pulse)
    control_paths = glob.glob(controls_folder+'/*.npz')
    for path in control_paths:
        name = os.path.basename(path)[:-4]
        figure_path = controls_figure_folder + f'/{name}.png'
        behaviour = np.load(path)
        single_responses = behaviour['single_responses']
        paired_responses = behaviour['paired_responses']

        # Single Pulses
        x = single_responses[0,:,:]
        y = single_responses[1,:,:]
        heading = single_responses[2,:,:]
        area = single_responses[3,:,:]
        valid_responses = []
        for r in range(x.shape[1]):
            if np.sum(area[:,r] == -1.0) < 5:
                large_response_size = np.abs((x[47,r] - x[60,r])) + np.abs((y[47,r] - y[70,r]))
                valid_responses.append(large_response_size)
        if len(valid_responses) > 0:
            mean_response = np.mean(np.array(valid_responses))
            control_mean_single.append(mean_response)

        # Paired Pulses
        x = paired_responses[0,:,:]
        y = paired_responses[1,:,:]
        heading = paired_responses[2,:,:]
        area = paired_responses[3,:,:]
        valid_responses = []
        for r in range(x.shape[1]):
            if np.sum(area[:,r] == -1.0) < 5:
                small_response_size = np.abs((x[18,r] - x[25,r])) + np.abs((y[19,r] - y[25,r]))
                if small_response_size < 10:
                    large_response_size = np.abs((x[47,r] - x[60,r])) + np.abs((y[49,r] - y[70,r]))
                    valid_responses.append(large_response_size)
        if len(valid_responses) > 0:
            mean_response = np.mean(np.array(valid_responses))
            control_mean_paired.append(mean_response)

    # Plot test responses
    test_paths = glob.glob(tests_folder+'/*.npz')
    for path in test_paths:
        name = os.path.basename(path)[:-4]
        figure_path = tests_figure_folder + f'/{name}.png'
        behaviour = np.load(path)
        single_responses = behaviour['single_responses']
        paired_responses = behaviour['paired_responses']

        # Single Pulses
        x = single_responses[0,:,:]
        y = single_responses[1,:,:]
        heading = single_responses[2,:,:]
        area = single_responses[3,:,:]
        valid_responses = []
        for r in range(x.shape[1]):
            if np.sum(area[:,r] == -1.0) < 5:
                large_response_size = np.abs((x[47,r] - x[60,r])) + np.abs((y[49,r] - y[70,r]))
                valid_responses.append(large_response_size)
        if len(valid_responses) > 0:
            mean_response = np.mean(np.array(valid_responses))
            test_mean_single.append(mean_response)

        # Paired Pulses
        x = paired_responses[0,:,:]
        y = paired_responses[1,:,:]
        heading = paired_responses[2,:,:]
        area = paired_responses[3,:,:]
        valid_responses = []
        for r in range(x.shape[1]):
            if np.sum(area[:,r] == -1.0) < 5:
                small_response_size = np.abs((x[18,r] - x[25,r])) + np.abs((y[19,r] - y[25,r]))
                if small_response_size < 10:
                    large_response_size = np.abs((x[47,r] - x[60,r])) + np.abs((y[49,r] - y[70,r]))
                    valid_responses.append(large_response_size)
        if len(valid_responses) > 0:
            mean_response = np.mean(np.array(valid_responses))
            test_mean_paired.append(mean_response)

# Plot summary figure
control_mean_single = np.array(control_mean_single)
control_mean_paired = np.array(control_mean_paired)
test_mean_single = np.array(test_mean_single)
test_mean_paired = np.array(test_mean_paired)
average_control_mean_single = np.mean(control_mean_single, axis=0)
average_control_mean_paired = np.mean(control_mean_paired, axis=0)
average_test_mean_single = np.mean(test_mean_single, axis=0)
average_test_mean_paired = np.mean(test_mean_paired, axis=0)

control_rate_single = np.mean(control_mean_single > 2)
control_rate_paired = np.mean(control_mean_paired > 2)
test_rate_single = np.mean(test_mean_single > 2)
test_rate_paired = np.mean(test_mean_paired > 2)

summary_path = experiment_analysis_folder + '/summary_alt.png'
fig = plt.figure(figsize=(10, 6))
plt.title(experiment)
plt.subplot(1,2,1)
plt.boxplot([control_mean_single, control_mean_paired])
plt.subplot(1,2,2)
plt.boxplot([test_mean_single, test_mean_paired])
plt.savefig(summary_path, dpi=180)
plt.close()

#FIN
