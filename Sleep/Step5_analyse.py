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
# Helper Functions
# ---------------------------------------------------------
def plot_individual_analysis(summary, name, figure_path):
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
    plt.savefig(figure_path, dpi=180)
    plt.cla()
    plt.close()
    return

# ---------------------------------------------------------

# Specify summary path
summary_path = base_path + "/Sumamry_Info.xlsx"

# Specify analysis parameters
individual_plots = True

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

# Analyse experiments
for experiment in experiments:
    plates, paths, controls, tests = MZU.parse_summary_Sleep(summary_path, experiment)
    # Report Plate-Path pairs
    for pl, pa in zip(plates, paths):
        print(pl, pa)

    # Set list of video paths
    path_list = paths

    # Analyse experiment
    for p, path in enumerate(path_list):
        # DEBUG (select a specific plate to work on)
        if (plates[p] != 3):
            continue

        # Print current path
        print(f"Working on path:{path}")

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

        # Check for valid lights file
        if not os.path.isfile(lights_path):
            print("No lights.csv file, please run previous step.")
            exit(-1)
        lights = np.genfromtxt(lights_path, delimiter=',')
        num_days = lights.shape[0]
        if (num_days < 2) or (lights[0,0] == "sunrise1"):
            print("Incorrect lights.csv file, please check previous step.")
            exit(-1)
            
        # Empty fish figures folder
        MZU.clear_folder(fish_figures_folder)

        # Compute BPM, sleep epochs, and other measures from individual bout data

        # Storage for average BPM
        control_wells = []
        control_summary = []
        control_epochs = []
        test_wells = []
        test_summary = []
        test_epochs = []

        # Analyse all control bouts
        control_paths = glob.glob(controls_folder+'/*.npz')
        for control_path in control_paths:
            name = os.path.basename(control_path)[:-4]
            well_number = int(name.split('_')[1])
            data = np.load(control_path)
            bouts = data['bouts']
            summary = MZB.compute_bouts_per_minute(bouts, 25)
            epochs = MZB.compute_sleep_epochs(bouts, lights, 25)
            control_wells.append(well_number)
            control_summary.append(summary)
            control_epochs.append(epochs)
            if individual_plots:
                figure_path = fish_figures_folder + f'/{name}.png'
                print(f"Plotting Control: {figure_path}")
                plot_individual_analysis(summary, name, figure_path)

        # Analyse all test bouts
        test_paths = glob.glob(tests_folder+'/*.npz')
        for test_path in test_paths:
            name = os.path.basename(test_path)[:-4]
            well_number = int(name.split('_')[1])
            data = np.load(test_path)
            bouts = data['bouts']
            summary = MZB.compute_bouts_per_minute(bouts, 25)
            epochs = MZB.compute_sleep_epochs(bouts, lights, 25)
            test_wells.append(well_number)
            test_summary.append(summary)
            test_epochs.append(epochs)
            if individual_plots:
                figure_path = fish_figures_folder + f'/{name}.png'
                print(f"Plotting Test: {figure_path}")
                plot_individual_analysis(summary, name, figure_path)

        # Determine longest fish trace
        num_control_fish = len(control_summary)
        num_test_fish = len(test_summary)
        num_fish = num_control_fish + num_test_fish
        max_length = 0
        for a in control_summary:
            this_length = len(a)
            if this_length > max_length:
                max_length = this_length
        for a in test_summary:
            this_length = len(a)
            if this_length > max_length:
                max_length = this_length
        control_summary_array = np.empty((num_control_fish, max_length, 6))
        test_summary_array = np.empty((num_test_fish, max_length, 6))
        control_summary_array[:] = np.nan
        test_summary_array[:] = np.nan

        # Fill arrays
        for i, a in enumerate(control_summary):
            this_length = len(a)
            control_summary_array[i, 0:this_length] = a
        control_mean_bpm = np.nanmean(control_summary_array[:,:,0], 0)
        for i, a in enumerate(test_summary):
            this_length = len(a)
            test_summary_array[i, 0:this_length] = a
        test_mean_bpm = np.nanmean(test_summary_array[:,:,0], 0)

        # Plot bout summary (per minute) comparison
        titles = ["BPM","Bout Durations","Bout Max Amps", "Bout Total Amps", "Bout Distances", "Bout Turns"]
        fig = plt.figure(figsize=(16, 9))
        plt.suptitle(f'Sleep: {path}')
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.title(titles[i])
            for night in lights:
                plt.axvspan(night[0], night[1], facecolor='0.2', alpha=0.15)
            plt.plot(np.nanmean(control_summary_array[:,:,i], 0), 'b', alpha=0.5)
            plt.plot(np.nanmean(test_summary_array[:,:,i], 0), 'r', alpha=0.5)
        summary_figure_path = figures_folder + f'/bout_summary.png'
        plt.savefig(summary_figure_path, dpi=180)
        plt.show()
        plt.close()

        # Plot per fish day/night sleep epochs
        fig = plt.figure(figsize=(5, 5))
        plt.subplot(1,2,1)
        for i, e in enumerate(control_epochs):
            plt.text(e[0], e[1], str(control_wells[i]))
            plt.plot(e[0], e[1], 'bo')
            print(i)
        for i, e in enumerate(test_epochs):
            plt.text(e[0], e[1], str(test_wells[i]))
            plt.plot(e[0], e[1], 'ro')
        plt.xlabel("#Night Sleep Epochs")
        plt.ylabel("#Day Sleep Epochs")
        plt.subplot(1,2,2)
        for i, e in enumerate(control_epochs):
            plt.text(e[2]/60, e[3]/60, str(control_wells[i]))
            plt.plot(e[2]/60, e[3]/60, 'bo')
            print(i)
        for i, e in enumerate(test_epochs):
            plt.text(e[2]/60, e[3]/60, str(test_wells[i]))
            plt.plot(e[2]/60, e[3]/60, 'ro')
        plt.xlabel("Night Sleep Duration (min)")
        plt.ylabel("Day Sleep Duration (min)")
        summary_figure_path = figures_folder + f'/sleep_summary.png'
        plt.savefig(summary_figure_path, dpi=180)
        plt.show()
        plt.close()

        # Do other analysis

#FIN
