# -*- coding: utf-8 -*-
"""
Analyse behaviour in a 96-well Sleep experiment

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
experiments = [#'Akap11', 
               #'Cacna1g', 
               'Gria3', 
               #'Grin2a',
               #'Hcn4',
               #'Herc1',
               #'Nr3c2',
               #'Sp4',
               #'Trio',
               #'Xpo7'
               ]

# Extract experiment
for experiment in experiments:
    plates, paths, controls, tests = MZU.parse_summary_Sleep(summary_path, experiment)

    # Set list of video paths
    path_list = paths

    # DEBUG - truncate path_list to only analyse one experiment
    path_list = [paths[1]]

    # Summarise experiment
    for p, path in enumerate(path_list):
        p = 1
        print(path)

        # Create Paths
        video_path = base_path + '/Sleep' + path
        output_folder = os.path.dirname(video_path) + '/analysis'
        plate_folder = output_folder + '/plate'
        responses_folder = output_folder + '/responses'
        controls_folder = responses_folder + '/controls'
        tests_folder = responses_folder + '/tests'
        figures_folder = output_folder + '/figures'
        fish_figures_folder = figures_folder + '/fish'

        # Empty fish figures folder
        MZU.clear_folder(fish_figures_folder)

        # Create summary structures
        controls_summary = []
        tests_summary = []

        # Analyse all control bouts
        control_paths = glob.glob(controls_folder+'/*.npz')
        for control_path in control_paths:
            name = os.path.basename(control_path)[:-4]
            well_number = int(name.split('_')[1])
            data = np.load(control_path)
            bouts = data['bouts']
            summary = MZB.compute_bouts_per_minute(bouts, 25)
            controls_summary.append(summary)
            fig = plt.figure(figsize=(10, 10))
            plt.suptitle(f'Sleep: {name}')
            plt.subplot(3,2,1)
            plt.plot(summary[:,0])
            plt.title('Bouts per Minute')
            plt.subplot(3,2,2)
            plt.plot(summary[:,1])
            plt.title('Durations')
            plt.subplot(3,2,3)
            plt.plot(summary[:,2])
            plt.title('Max Motion')
            plt.subplot(3,2,4)
            plt.plot(summary[:,3])
            plt.title('Total Motion')
            plt.subplot(3,2,5)
            plt.plot(summary[:,4])
            plt.title('Distance')
            plt.subplot(3,2,6)
            plt.plot(summary[:,5])
            plt.title('Turning')
            control_figure_path = fish_figures_folder + f'/{name}.png'
            plt.savefig(control_figure_path, dpi=180)
            plt.cla()
            plt.close()

        # Analyse all test bouts
        test_paths = glob.glob(tests_folder+'/*.npz')
        for test_path in test_paths:
            name = os.path.basename(test_path)[:-4]
            well_number = int(name.split('_')[1])
            data = np.load(test_path)
            bouts = data['bouts']
            summary = MZB.compute_bouts_per_minute(bouts, 25)
            tests_summary.append(summary)
            fig = plt.figure(figsize=(10, 10))
            plt.suptitle(f'Sleep: {name}')
            plt.subplot(3,2,1)
            plt.plot(summary[:,0])
            plt.title('Bouts per Minute')
            plt.subplot(3,2,2)
            plt.plot(summary[:,1])
            plt.title('Durations')
            plt.subplot(3,2,3)
            plt.plot(summary[:,2])
            plt.title('Max Motion')
            plt.subplot(3,2,4)
            plt.plot(summary[:,3])
            plt.title('Total Motion')
            plt.subplot(3,2,5)
            plt.plot(summary[:,4])
            plt.title('Distance')
            plt.subplot(3,2,6)
            plt.plot(summary[:,5])
            plt.title('Turning')
            test_figure_path = fish_figures_folder + f'/{name}.png'
            plt.savefig(test_figure_path, dpi=180)
            plt.cla()
            plt.close()

    # Find smallest summary
    summary_lengths = []
    for cs in controls_summary:
        summary_lengths.append(len(cs))
    for ts in tests_summary:
        summary_lengths.append(len(ts))
    summary_length = min(summary_lengths)

    # Assemble summary data
    controls_summary_array = np.zeros((summary_length, 6, len(controls[p])))
    tests_summary_array = np.zeros((summary_length, 6, len(tests[p])))
    for s, summary in enumerate(controls_summary):
        controls_summary_array[:,:,s]  = summary[:summary_length,:]
    for s, summary in enumerate(tests_summary):
        tests_summary_array[:,:,s]  = summary[:summary_length,:]

    # Plot control means
    fig = plt.figure(figsize=(10, 10))
    plt.suptitle(f'Sleep: {path}')
    plt.subplot(3,2,1)
    plt.plot(controls_summary_array[:,0,:], 'b', alpha=0.01)
    plt.plot(np.nanmean(controls_summary_array[:,0,:], axis=1), 'k', alpha=1.00)
    plt.ylim([0,100])
    plt.title('Bouts per Minute')
    plt.subplot(3,2,2)
    plt.plot(controls_summary_array[:,1,:], 'b', alpha=0.01)
    plt.plot(np.nanmean(controls_summary_array[:,1,:], axis=1), 'k', alpha=1.00)
    plt.ylim([0,50])
    plt.title('Durations')
    plt.subplot(3,2,3)
    plt.plot(controls_summary_array[:,2,:], 'b', alpha=0.01)
    plt.plot(np.nanmean(controls_summary_array[:,2,:], axis=1), 'k', alpha=1.00)
    plt.ylim([0,5000])
    plt.title('Max Motion')
    plt.subplot(3,2,4)
    plt.plot(controls_summary_array[:,3,:], 'b', alpha=0.01)
    plt.plot(np.nanmean(controls_summary_array[:,3,:], axis=1), 'k', alpha=1.00)
    plt.ylim([0,30000])
    plt.title('Total Motion')
    plt.subplot(3,2,5)
    plt.plot(controls_summary_array[:,4,:], 'b', alpha=0.01)
    plt.plot(np.nanmean(controls_summary_array[:,4,:], axis=1), 'k', alpha=1.00)
    plt.ylim([0,50])
    plt.title('Distance')
    plt.subplot(3,2,6)
    plt.plot(controls_summary_array[:,5,:], 'b', alpha=0.01)
    plt.plot(np.nanmean(controls_summary_array[:,5,:], axis=1), 'k', alpha=1.00)
    plt.ylim([-180,180])
    plt.title('Turning')
    summary_figure_path = figures_folder + f'/controls_summary.png'
    plt.savefig(summary_figure_path, dpi=180)
    plt.cla()

    # Plot test means
    fig = plt.figure(figsize=(10, 10))
    plt.suptitle(f'Sleep: {path}')
    plt.subplot(3,2,1)
    plt.plot(tests_summary_array[:,0,:], 'b', alpha=0.01)
    plt.plot(np.nanmean(tests_summary_array[:,0,:], axis=1), 'k', alpha=1.00)
    plt.ylim([0,100])
    plt.title('Bouts per Minute')
    plt.subplot(3,2,2)
    plt.plot(tests_summary_array[:,1,:], 'b', alpha=0.01)
    plt.plot(np.nanmean(tests_summary_array[:,1,:], axis=1), 'k', alpha=1.00)
    plt.ylim([0,50])
    plt.title('Durations')
    plt.subplot(3,2,3)
    plt.plot(tests_summary_array[:,2,:], 'b', alpha=0.01)
    plt.plot(np.nanmean(tests_summary_array[:,2,:], axis=1), 'k', alpha=1.00)
    plt.ylim([0,5000])
    plt.title('Max Motion')
    plt.subplot(3,2,4)
    plt.plot(tests_summary_array[:,3,:], 'b', alpha=0.01)
    plt.plot(np.nanmean(tests_summary_array[:,3,:], axis=1), 'k', alpha=1.00)
    plt.ylim([0,30000])
    plt.title('Total Motion')
    plt.subplot(3,2,5)
    plt.plot(tests_summary_array[:,4,:], 'b', alpha=0.01)
    plt.plot(np.nanmean(tests_summary_array[:,4,:], axis=1), 'k', alpha=1.00)
    plt.ylim([0,50])
    plt.title('Distance')
    plt.subplot(3,2,6)
    plt.plot(tests_summary_array[:,5,:], 'b', alpha=0.01)
    plt.plot(np.nanmean(tests_summary_array[:,5,:], axis=1), 'k', alpha=1.00)
    plt.ylim([-180,180])
    plt.title('Turning')
    summary_figure_path = figures_folder + f'/tests_summary.png'
    plt.savefig(summary_figure_path, dpi=180)
    plt.cla()

#FIN
