# -*- coding: utf-8 -*-
"""
Summarise behaviour (bout analysis) in a 96-well Sleep experiment

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
    # Report Plate-Path pairs
    for pl, pa in zip(plates, paths):
        print(pl, pa)

    # Set list of video paths
    path_list = paths

    # Summarise experiment
    for p, path in enumerate(path_list):
        # DEBUG (select a specific plate to work on)
        if (plates[p] != 16):
            continue

        # Print current path
        print(f"Working on path:{path}")

        # Create Paths
        video_path = base_path + '/Sleep' + path
        output_folder = os.path.dirname(video_path) + '/analysis'
        experiment_name = path.split('/')[1]
        experiment_folder = base_path + '/Sleep/' + experiment_name
        plate_folder = output_folder + '/plate'
        responses_folder = output_folder + '/responses'
        controls_folder = responses_folder + '/controls'
        tests_folder = responses_folder + '/tests'
        figures_folder = output_folder + '/figures'

        # Empty responses folder
        MZU.clear_folder(responses_folder)
        MZU.create_folder(controls_folder)
        MZU.create_folder(tests_folder)

        # Empty figures folder
        MZU.clear_folder(figures_folder)

        # Create plate structure
        name = path.split('/')[-1][:-4]
        plate = MZP.Plate(name)

        # Load plate in chunks
        plate_paths = sorted(glob.glob(plate_folder + '/*.npz'), key=os.path.getmtime)
        all_intensity = np.empty(0, dtype=np.float32)
        all_bouts = np.empty(96, object)
        for w in range(96):
            all_bouts[w] = np.empty((0,8))
        for plate_path in plate_paths:
            frame_range = plate_path[:-4].split('_')[-2:]
            start_frame = int(frame_range[0])
            end_frame = int(frame_range[1])

            # Load plate
            print(f'Loading plate data chunk...{start_frame} to {end_frame}')
            try:
                plate.load(output_folder, start_frame, end_frame)
            except ValueError:
                print(f"Corrupt plate file {plate_path}")

            # Extract and store all bouts
            for w in range(96):
                fish = plate.wells[w]
                behaviour = MZU.extract_behaviour(plate, w)
                bouts = MZB.extract_bouts_array(behaviour, frame_offset=start_frame)
                all_bouts[w] = np.vstack((all_bouts[w], bouts))
                
            # Extract and store intensity
            all_intensity = np.hstack((all_intensity, plate.intensity))

            # Clear plate structure
            plate.clear()

        # Load intensity
        num_frames = len(all_intensity)
        print(num_frames)

        # Plot intensity
        fig = plt.figure(figsize=(10, 4))
        plt.title('Background Intensity Detection')
        plt.plot(all_intensity)
        plt.savefig(figures_folder + '/intensity.png', dpi=180)
        plt.cla()
        plt.close()
        
        # Save all control bouts
        for c in controls[p]:
            control_path = controls_folder + f'/control_{c}_plate_{plates[p]}.npz'
            np.savez(control_path, bouts=all_bouts[c-1])

        # Save all test bouts
        for t in tests[p]:
            test_path = tests_folder + f'/test_{t}_plate_{plates[p]}.npz'
            np.savez(test_path, bouts=all_bouts[t-1])
#FIN
