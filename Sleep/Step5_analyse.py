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

#----------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def plot_individual_analysis(summary, timecourse, lights, name, figure_path):
    fig = plt.figure(figsize=(12, 4))
    plt.suptitle(f'Sleep: {name}')
    
    ax = plt.subplot(2,4,1)
    for night in lights:
        plt.axvspan(night[0], night[1], facecolor='0.2', alpha=0.15)
    plt.plot(summary[:,0])
    plt.title('Bouts per Minute')
    x_axis = ax.axes.get_xaxis()
    x_axis.set_visible(False)
    
    ax = plt.subplot(2,4,2)
    for night in lights:
        plt.axvspan(night[0], night[1], facecolor='0.2', alpha=0.15)
    plt.plot(summary[:,1])
    plt.title('Durations')
    x_axis = ax.axes.get_xaxis()
    x_axis.set_visible(False)
    
    ax = plt.subplot(2,4,3)
    for night in lights:
        plt.axvspan(night[0], night[1], facecolor='0.2', alpha=0.15)
    plt.plot(summary[:,2])
    plt.title('Max Motion')
    x_axis = ax.axes.get_xaxis()
    x_axis.set_visible(False)
   
    plt.subplot(2,4,4)
    for night in lights:
        plt.axvspan(night[0]/10, night[1]/10, facecolor='0.2', alpha=0.15)
    plt.plot(timecourse)
    plt.title('Timecourse (secs asleep/10 min)')
    
    plt.subplot(2,4,5)
    for night in lights:
        plt.axvspan(night[0], night[1], facecolor='0.2', alpha=0.15)
    plt.plot(summary[:,3])
    plt.title('Total Motion')
    
    plt.subplot(2,4,6)
    for night in lights:
        plt.axvspan(night[0], night[1], facecolor='0.2', alpha=0.15)
    plt.plot(summary[:,4])
    plt.title('Distance')
    
    plt.subplot(2,4,7)
    for night in lights:
        plt.axvspan(night[0], night[1], facecolor='0.2', alpha=0.15)
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
show_summary_plots = True

# Specify experiment abbreviation
experiments = [#'Akap11', 
               #'Cacna1g', 
               #'Gria3', 
               'Grin2a',
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
        if (plates[p] != 25):
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
        lights = np.genfromtxt(lights_path, delimiter=',', encoding="utf-8-sig", dtype=int)
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
        control_timecourses = []
        test_wells = []
        test_summary = []
        test_epochs = []
        test_timecourses = []

        # Analyse all control bouts
        control_paths = glob.glob(controls_folder+'/*.npz')
        for control_path in control_paths:
            name = os.path.basename(control_path)[:-4]
            well_number = int(name.split('_')[1])
            data = np.load(control_path)
            bouts = data['bouts']
            summary = MZB.compute_bouts_per_minute(bouts, 25)
            epochs = MZB.compute_sleep_epochs(bouts, lights, 25)
            timecourse = MZB.compute_sleep_timecourse(bouts, lights, 25)
            control_wells.append(well_number)
            control_summary.append(summary)
            control_epochs.append(epochs)
            control_timecourses.append(timecourse)
            if individual_plots:
                figure_path = fish_figures_folder + f'/{name}.png'
                print(f"Plotting Control: {figure_path}")
                plot_individual_analysis(summary, timecourse, lights, name, figure_path)

        # Analyse all test bouts
        test_paths = glob.glob(tests_folder+'/*.npz')
        for test_path in test_paths:
            name = os.path.basename(test_path)[:-4]
            well_number = int(name.split('_')[1])
            data = np.load(test_path)
            bouts = data['bouts']
            summary = MZB.compute_bouts_per_minute(bouts, 25)
            epochs = MZB.compute_sleep_epochs(bouts, lights, 25)
            timecourse = MZB.compute_sleep_timecourse(bouts, lights, 25)
            test_wells.append(well_number)
            test_summary.append(summary)
            test_epochs.append(epochs)
            test_timecourses.append(timecourse)
            if individual_plots:
                figure_path = fish_figures_folder + f'/{name}.png'
                print(f"Plotting Test: {figure_path}")
                plot_individual_analysis(summary, timecourse, lights, name, figure_path)

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

        # Fill summary arrays
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
        if show_summary_plots:
            plt.show()
        plt.close()
        summary_data_path = figures_folder + f'/bout_summary.npz'
        np.savez(summary_data_path, control_summary_array=control_summary_array, test_summary_array=test_summary_array)

        # Create timecourse arrays
        control_timecourse_array = np.array(control_timecourses)
        test_timecourse_array = np.array(test_timecourses)

        # Plot sleep timecourse summary (per 10 minute) comparison
        fig = plt.figure(figsize=(9, 6))
        for night in lights:
            plt.axvspan(night[0]/10, night[1]/10, facecolor='0.2', alpha=0.15)
        #plt.plot(control_timecourse_array.T, 'b', alpha=0.01)
        #plt.plot(test_timecourse_array.T, 'r', alpha=0.01)
        plt.plot(np.nanmean(control_timecourse_array, 0), 'b', alpha=0.5)
        plt.plot(np.nanmean(test_timecourse_array, 0), 'r', alpha=0.5)
        plt.title('Sleep Timecourse')
        plt.ylabel('Seconds asleep per 10 minutes')
        summary_figure_path = figures_folder + f'/timecourse_summary.png'
        plt.savefig(summary_figure_path, dpi=180)
        if show_summary_plots:
            plt.show()
        plt.close()
        summary_data_path = figures_folder + f'/timecourse_summary.npz'
        np.savez(summary_data_path, control_timecourse_array=control_timecourse_array, test_timecourse_array=test_timecourse_array)

        # Plot per fish day/night sleep epochs - controls vs tests
        control_epochs_array = np.array(control_epochs)
        test_epochs_array = np.array(test_epochs)

        fig = plt.figure(figsize=(10, 5))
        night_data = [control_epochs_array[:,0], test_epochs_array[:,0]]
        day_data = [control_epochs_array[:,1], test_epochs_array[:,1]]
        plt.subplot(2,2,1)
        plt.title('Night Sleep')
        sns.stripplot(data=night_data)
        sns.pointplot(data=night_data, linestyle="none", errorbar=('ci', 95), marker=".", color="k", markersize=10, markeredgewidth=1)
        plt.ylabel("#Sleep Epochs")
        plt.xticks([0,1], ["Scrambled", f"{experiment}"])
        plt.subplot(2,2,2)
        plt.title('Day Sleep')
        sns.stripplot(data=day_data)
        sns.pointplot(data=day_data, linestyle="none", errorbar=('ci', 95), marker=".", color="k", markersize=10, markeredgewidth=1)
        plt.ylabel("#Sleep Epochs")
        plt.xticks([0,1], ["Scrambled", f"{experiment}"])

        night_data = [control_epochs_array[:,2], test_epochs_array[:,2]]
        day_data = [control_epochs_array[:,3], test_epochs_array[:,3]]
        plt.subplot(2,2,3)
        sns.stripplot(data=night_data)
        sns.pointplot(data=night_data, linestyle="none", errorbar=('ci', 95), marker=".", color="k", markersize=10, markeredgewidth=1)
        plt.ylabel("Sleep Duration (min)")
        plt.xticks([0,1], ["Scrambled", f"{experiment}"])
        plt.subplot(2,2,4)
        sns.stripplot(data=day_data)
        sns.pointplot(data=day_data, linestyle="none", errorbar=('ci', 95), marker=".", color="k", markersize=10, markeredgewidth=1)
        plt.ylabel("Sleep Duration (min)")
        plt.xticks([0,1], ["Scrambled", f"{experiment}"])

        summary_figure_path = figures_folder + f'/sleep_summary.png'
        plt.savefig(summary_figure_path, dpi=180)
        if show_summary_plots:
            plt.show()
        plt.close()
        summary_data_path = figures_folder + f'/sleep_summary.npz'
        np.savez(summary_data_path, control_epochs_array=control_epochs_array, test_epochs_array=test_epochs_array)

        # Do other analysis

#FIN
