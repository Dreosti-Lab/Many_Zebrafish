# -*- coding: utf-8 -*-
"""
Combined analysed behaviour from mulitple 96-well Sleep experiments

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
import seaborn as sns

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

# Specify summary path
summary_path = base_path + "/Sumamry_Info.xlsx"

# Specify analysis parameters
individual_plots = True

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
    # Report Plate-Path pairs
    for pl, pa in zip(plates, paths):
        print(pl, pa)

    # Set list of video paths
    path_list = paths

    # Specify experiment folder
    experiment_analysis_folder = base_path + '/Sleep' + '/_analysis'

    # Create empty containers
    all_control_bpm_night = np.zeros(0)
    all_control_bpm_day = np.zeros(0)
    all_test_bpm_night = np.zeros(0)
    all_test_bpm_day = np.zeros(0)
    all_control_epochs_night = np.zeros(0)
    all_control_epochs_day = np.zeros(0)
    all_test_epochs_night = np.zeros(0)
    all_test_epochs_day = np.zeros(0)
    all_control_durations_night = np.zeros(0)
    all_control_durations_day = np.zeros(0)
    all_test_durations_night = np.zeros(0)
    all_test_durations_day = np.zeros(0)

    # Analyse experiment
    for p, path in enumerate(path_list):
        # DEBUG (select a specific plate to EXCLUDE)
        if (plates[p] == 39):
            continue

        # Print current path
        print(f"Working on path:{path}")

        # Create Paths
        video_path = base_path + '/Sleep' + path
        output_folder = os.path.dirname(video_path) + '/analysis'
        lights_path = output_folder + '/lights.csv'
        figures_folder = output_folder + '/figures'

        # Check for valid lights file
        if not os.path.isfile(lights_path):
            print("No lights.csv file, please run previous step.")
            exit(-1)
        lights = np.genfromtxt(lights_path, delimiter=',', encoding="utf-8-sig", dtype=int)
        num_days = lights.shape[0]
        if (num_days < 2) or (lights[0,0] == "sunrise1"):
            print("Incorrect lights.csv file, please check previous step.")
            exit(-1)
        
        # Load (bout) summary data
        bout_summary_path = figures_folder + f'/bout_summary.npz'
        bout_summary = np.load(bout_summary_path)
        control_summary_array = bout_summary['control_summary_array']
        test_summary_array = bout_summary['test_summary_array']

        # Build day(0)/night(1) per minute filter
        num_minutes = control_summary_array.shape[1]
        day_night_filter = np.zeros(num_minutes, dtype=bool)
        for night in lights:
            day_night_filter[night[0]:night[1]] = True
        num_night_minutes = np.sum(day_night_filter)
        num_day_minutes = num_minutes - num_night_minutes

        # Compte day/night bouts per minute (activity) for controls vs tests and normalize (percentage of controls)
        control_bpm_night = np.nanmean(control_summary_array[:,day_night_filter,0], axis=1)
        control_bpm_day = np.nanmean(control_summary_array[:,np.logical_not(day_night_filter),0], axis=1)
        test_bpm_night = np.nanmean(test_summary_array[:,day_night_filter,0], axis=1)
        test_bpm_day = np.nanmean(test_summary_array[:,np.logical_not(day_night_filter),0], axis=1)

        mean_control_bpm_night = np.mean(control_bpm_night)
        control_bpm_night_norm = 100.0 * (control_bpm_night - mean_control_bpm_night) / mean_control_bpm_night
        test_bpm_night_norm = 100.0 * (test_bpm_night - mean_control_bpm_night) / mean_control_bpm_night

        mean_control_bpm_day = np.mean(control_bpm_day)
        control_bpm_day_norm = 100.0 * (control_bpm_day - mean_control_bpm_day) / mean_control_bpm_day
        test_bpm_day_norm = 100.0 * (test_bpm_day - mean_control_bpm_day) / mean_control_bpm_day

        # Load (sleep) epochs data
        sleep_summary_path = figures_folder + f'/sleep_summary.npz'
        sleep_summary = np.load(sleep_summary_path)
        control_epochs_array = sleep_summary['control_epochs_array']
        test_epochs_array = sleep_summary['test_epochs_array']

        # Extract epoch count and sleep duration and normalize (percentage of controls)
        control_epochs_night = control_epochs_array[:,0]
        test_epochs_night = test_epochs_array[:,0]
        mean_control_epochs_night = np.mean(control_epochs_night)
        control_epochs_night_norm = 100.0 * (control_epochs_night - mean_control_epochs_night) / mean_control_epochs_night
        test_epochs_night_norm = 100.0 * (test_epochs_night - mean_control_epochs_night) / mean_control_epochs_night

        control_epochs_day = control_epochs_array[:,1]
        test_epochs_day = test_epochs_array[:,1]
        mean_control_epochs_day = np.mean(control_epochs_day)
        control_epochs_day_norm = 100.0 * (control_epochs_day - mean_control_epochs_day) / mean_control_epochs_day
        test_epochs_day_norm = 100.0 * (test_epochs_day - mean_control_epochs_day) / mean_control_epochs_day

        control_durations_night = control_epochs_array[:,2]
        test_durations_night = test_epochs_array[:,2]
        mean_control_durations_night = np.mean(control_durations_night)
        control_durations_night_norm = 100.0 * (control_durations_night - mean_control_durations_night) / mean_control_durations_night
        test_durations_night_norm = 100.0 * (test_durations_night - mean_control_durations_night) / mean_control_durations_night

        control_durations_day = control_epochs_array[:,3]
        test_durations_day = test_epochs_array[:,3]
        mean_control_durations_day = np.mean(control_durations_day)
        control_durations_day_norm = 100.0 * (control_durations_day - mean_control_durations_day) / mean_control_durations_day
        test_durations_day_norm = 100.0 * (test_durations_day - mean_control_durations_day) / mean_control_durations_day

        # Append
        all_control_bpm_night = np.hstack((all_control_bpm_night, control_bpm_night_norm))
        all_control_bpm_day = np.hstack((all_control_bpm_day, control_bpm_day_norm))
        all_test_bpm_night = np.hstack((all_test_bpm_night, test_bpm_night_norm))
        all_test_bpm_day = np.hstack((all_test_bpm_day, test_bpm_day_norm))
        all_control_epochs_night = np.hstack((all_control_epochs_night, control_epochs_night_norm))
        all_control_epochs_day = np.hstack((all_control_epochs_day, control_epochs_day_norm))
        all_test_epochs_night = np.hstack((all_test_epochs_night, test_epochs_night_norm))
        all_test_epochs_day = np.hstack((all_test_epochs_day, test_epochs_day_norm))
        all_control_durations_night = np.hstack((all_control_durations_night, control_durations_night_norm))
        all_control_durations_day = np.hstack((all_control_durations_day, control_durations_day_norm))
        all_test_durations_night = np.hstack((all_test_durations_night, test_durations_night_norm))
        all_test_durations_day = np.hstack((all_test_durations_day, test_durations_day_norm))

    # Summary (combined) plots ACTIVITY
    fig = plt.figure(figsize=(10, 5))
    night_data = [all_control_bpm_night, all_test_bpm_night]
    day_data = [all_control_bpm_day, all_test_bpm_day]
    plt.subplot(1,2,1)
    plt.title('Night Activity')
    sns.stripplot(data=night_data, alpha=0.25)
    sns.pointplot(data=night_data, linestyle="none", errorbar=None, marker="_", color="k", markersize=20, markeredgewidth=3)
    plt.ylabel("Bouts per Minute")
    plt.xticks([0,1], ["Scrambled", f"{experiment}"])
    plt.subplot(1,2,2)
    plt.title('Day Activity')
    sns.stripplot(data=day_data, alpha=0.25)
    sns.pointplot(data=day_data, linestyle="none", errorbar=None, marker="_", color="k", markersize=20, markeredgewidth=3)
    plt.ylabel("Bouts per Minute")
    plt.xticks([0,1], ["Scrambled", f"{experiment}"])

    combined_summary_figure_path = experiment_analysis_folder + f'/{experiment}_activty_summary.png'
    plt.savefig(combined_summary_figure_path, dpi=180)
    plt.show()
    plt.close()

    # Summary (combined) plots SLEEP
    fig = plt.figure(figsize=(10, 5))
    night_data = [all_control_epochs_night, all_test_epochs_night]
    day_data = [all_control_epochs_day, all_test_epochs_day]
    plt.subplot(2,2,1)
    plt.title('Night Sleep')
    sns.stripplot(data=night_data, alpha=0.25)
    sns.pointplot(data=night_data, linestyle="none", errorbar=None, marker="_", color="k", markersize=20, markeredgewidth=3)
    plt.ylabel("#Sleep Epochs")
    plt.xticks([0,1], ["Scrambled", f"{experiment}"])
    plt.subplot(2,2,2)
    plt.title('Day Sleep')
    sns.stripplot(data=day_data, alpha=0.25)
    sns.pointplot(data=day_data, linestyle="none", errorbar=None, marker="_", color="k", markersize=20, markeredgewidth=3)
    plt.ylabel("#Sleep Epochs")
    plt.xticks([0,1], ["Scrambled", f"{experiment}"])

    night_data = [all_control_durations_night, all_test_durations_night]
    day_data = [all_control_durations_day, all_test_durations_day]
    plt.subplot(2,2,3)
    sns.stripplot(data=night_data, alpha=0.25)
    sns.pointplot(data=night_data, linestyle="none", errorbar=None, marker="_", color="k", markersize=20, markeredgewidth=3)
    plt.ylabel("Sleep Duration (min)")
    plt.xticks([0,1], ["Scrambled", f"{experiment}"])
    plt.subplot(2,2,4)
    sns.stripplot(data=day_data, alpha=0.25)
    sns.pointplot(data=day_data, linestyle="none", errorbar=None, marker="_", color="k", markersize=20, markeredgewidth=3)
    plt.ylabel("Sleep Duration (min)")
    plt.xticks([0,1], ["Scrambled", f"{experiment}"])

    combined_summary_figure_path = experiment_analysis_folder + f'/{experiment}_sleep_summary.png'
    plt.savefig(combined_summary_figure_path, dpi=180)
    plt.show()
    plt.close()

#FIN
