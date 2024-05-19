# -*- coding: utf-8 -*-
"""
Extract light timing in a 96-well Sleep experiment

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

        # Compute BPM, sleep epochs, and other measures from individual bout data

        # Storage for average BPM
        all_bpm = []

        # Analyse all control bouts
        control_paths = glob.glob(controls_folder+'/*.npz')
        for control_path in control_paths:
            name = os.path.basename(control_path)[:-4]
            well_number = int(name.split('_')[1])
            data = np.load(control_path)
            bouts = data['bouts']
            summary = MZB.compute_bouts_per_minute(bouts, 25)
            all_bpm.append(summary)
            print(control_path)

        # Analyse all test bouts
        test_paths = glob.glob(tests_folder+'/*.npz')
        for test_path in test_paths:
            name = os.path.basename(test_path)[:-4]
            well_number = int(name.split('_')[1])
            data = np.load(test_path)
            bouts = data['bouts']
            summary = MZB.compute_bouts_per_minute(bouts, 25)
            all_bpm.append(summary)
            print(test_path)

        # Determine longest fish trace
        num_fish = len(all_bpm)
        max_length = 0
        for a in all_bpm:
            this_length = len(a)
            if this_length > max_length:
                max_length = this_length
        all_bpm_array = np.zeros((num_fish, max_length))

        # Fill array
        for i, a in enumerate(all_bpm):
            this_length = len(a)
            all_bpm_array[i, 0:this_length] = a[:,0]
        mean_bpm = np.mean(all_bpm_array, 0)

        # Create template "lights.csv" (if it does not exist)
        if not os.path.isfile(lights_path):
            f = open(lights_path, 'w')
            f.write("sunset1,sunrise1\n")
            f.write("sunset2,sunrise2\n")
            f.write("sunset3,sunrise3\n")
            f.close()

        # Plot BPM average (select sunrise, sunsets)
        plt.figure()
        plt.plot(mean_bpm)
        plt.show()

#FIN
