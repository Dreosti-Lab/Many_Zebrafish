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
from scipy import stats
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

# Specify analysis parameters
individual_plots = False

# Specify experiment abbreviation
experiments = [#'Akap11', 
               #'Cacna1g', 
               #'Gria3', 
               #'Grin2a',
               #'Hcn4',
               #'Herc1',
               #'Nr3c2',
               #'Sp4',
               'Trio',
               #'Xpo7'
               ]

# Analyse experiments
for experiment in experiments:
    plates, paths, controls, tests = MZU.parse_summary_Sleep(summary_path, experiment)

    # Set list of video paths
    path_list = paths

    # Bulk structures
    all_controls_mean_summary = []
    all_tests_mean_summary = []
    all_controls_mean_epochs = []
    all_tests_mean_epochs = []

    # Analyse experiment
    for p, path in enumerate(path_list):
        print(path)

        # DEBUG
        if (plates[p] != 22):
            continue

        # Create Paths
        video_path = base_path + '/Sleep' + path
        output_folder = os.path.dirname(video_path) + '/analysis'
        lights_path = output_folder + '/lights.csv'
        plate_folder = output_folder + '/plate'
        responses_folder = output_folder + '/responses'
        controls_folder = responses_folder + '/controls'
        tests_folder = responses_folder + '/tests'
        figures_folder = output_folder + '/figures'
        fish_figures_folder = figures_folder + '/fish'
        experiment_analysis_folder = base_path + '/Sleep' + '/_analysis'

        # Empty fish figures folder
        MZU.clear_folder(fish_figures_folder)

        # Create experiment analysis folder
        MZU.create_folder(experiment_analysis_folder)

        # Load lights
        lights = np.genfromtxt(lights_path, delimiter=',')
        num_days = lights.shape[0]

        # Create summary structures
        controls_summary = []
        tests_summary = []
        controls_epochs = []
        tests_epochs = []

        # Analyse all control bouts
        control_paths = glob.glob(controls_folder+'/*.npz')
        for control_path in control_paths:
            name = os.path.basename(control_path)[:-4]
            well_number = int(name.split('_')[1])
            data = np.load(control_path)
            bouts = data['bouts']
            summary = MZB.compute_bouts_per_minute(bouts, 25)
            controls_summary.append(summary)
            epochs = MZB.compute_sleep_epochs(bouts, lights, 25)
            controls_epochs.append(epochs)
            if individual_plots:
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
            epochs = MZB.compute_sleep_epochs(bouts, lights, 25)
            tests_epochs.append(epochs)
            if individual_plots:
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

        # Assemble epoch data
        controls_epochs_array = np.array(controls_epochs)
        tests_epochs_array = np.array(tests_epochs)

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
        #plt.show()
        plt.cla()
        plt.close()

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
        #plt.show()
        plt.cla()
        plt.close()

        # Bulk assemble
        shift = int(lights[0,0] - 200)
        end = 3650
        all_controls_mean_summary.append(np.nanmean(controls_summary_array[shift:(shift+3600),:,:], axis=2))
        all_tests_mean_summary.append(np.nanmean(tests_summary_array[shift:(shift+3600),:,:], axis=2))
        all_controls_mean_epochs.append(np.nanmean(controls_epochs_array, axis=0))
        all_tests_mean_epochs.append(np.nanmean(tests_epochs_array, axis=0))

    # Experiment stats
    all_controls_mean_epochs_array = np.array(all_controls_mean_epochs)
    all_tests_mean_epochs_array = np.array(all_tests_mean_epochs)

    controls_mean_night = all_controls_mean_epochs_array[:,0]
    controls_mean_day = all_controls_mean_epochs_array[:,1]
    tests_mean_night = all_tests_mean_epochs_array[:,0]
    tests_mean_day = all_tests_mean_epochs_array[:,1]

    avg_controls_mean_night = np.mean(controls_mean_night)
    avg_controls_mean_day = np.mean(controls_mean_day)
    avg_tests_mean_night = np.mean(tests_mean_night)
    avg_tests_mean_day = np.mean(tests_mean_day)

    std_controls_mean_night = np.std(controls_mean_night)
    std_controls_mean_day = np.std(controls_mean_day)
    std_tests_mean_night = np.std(tests_mean_night)
    std_tests_mean_day = np.std(tests_mean_day)

    p_value_night_v_day_controls = stats.ttest_rel(controls_mean_night, controls_mean_day)[1]
    p_value_night_v_day_tests = stats.ttest_rel(tests_mean_night, tests_mean_day)[1]
    p_value_controls_v_tests_night = stats.ttest_rel(controls_mean_night, tests_mean_night)[1]
    p_value_controls_v_tests_day = stats.ttest_rel(controls_mean_day, tests_mean_day)[1]
    summary_stats = f'C-NvD: {p_value_night_v_day_controls:.3f}, T-NvD: {p_value_night_v_day_tests:.3f}, N-CvT: {p_value_controls_v_tests_night:.3f}, D-CvT: {p_value_controls_v_tests_day:.3f}'

    # Experiment summary
    all_controls_mean_summary_array = np.array(all_controls_mean_summary)
    all_tests_mean_summary_array = np.array(all_tests_mean_summary)
    fig = plt.figure(figsize=(10, 10))
    plt.suptitle(f'Sleep: {experiment}')
    plt.subplot(2,1,1)
    plt.title(summary_stats)
    plt.xticks([1,2,3,4], ['Control (N)', 'Tests (N)', 'Control (D)', 'Tests (D)'], rotation='horizontal')
    plt.ylabel('Number of Sleep Epochs')
    plt.bar(1, avg_controls_mean_night, color = (0.3,0.3,1))
    plt.bar(2, avg_tests_mean_night, color = (1,0.3,0.3))
    plt.bar(3, avg_controls_mean_day, color = (0.3,0.3,1))
    plt.bar(4, avg_tests_mean_day, color = (1,0.3,0.3))
    plt.errorbar([1,2,3,4], [avg_controls_mean_night, avg_tests_mean_night, avg_controls_mean_day, avg_tests_mean_day], yerr=[std_controls_mean_night, std_tests_mean_night, std_controls_mean_day, std_tests_mean_day], fmt='o', color='black')
    plt.subplot(2,1,2)
    plt.plot(all_controls_mean_summary_array[:,:,0].T, 'b', alpha=0.1)
    plt.plot(np.mean(all_controls_mean_summary_array[:,:,0], axis=0), 'b', linewidth=2, alpha=1.0, label=f'Controls')
    plt.plot(all_tests_mean_summary_array[:,:,0].T, 'r', alpha=0.1)
    plt.plot(np.mean(all_tests_mean_summary_array[:,:,0], axis=0), 'r', linewidth=2, alpha=1.0, label=f'{experiment}')
    plt.legend()
    experiment_summary_figure_path = experiment_analysis_folder + f'/{experiment}_BPM_summary.png'
    plt.savefig(experiment_summary_figure_path, dpi=180)
    experiment_summary_figure_path = experiment_analysis_folder + f'/{experiment}_BPM_summary.svg'
    plt.savefig(experiment_summary_figure_path, dpi=180)
    plt.show()
    plt.cla()
    plt.close()

#FIN
