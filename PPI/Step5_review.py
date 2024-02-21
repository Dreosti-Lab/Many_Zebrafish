# -*- coding: utf-8 -*-
"""
Review behaviour reponses in a 96-well PPI experiment

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
import cv2

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
experiment = 'Akap11'
#experiment = 'Cacna1g'
#experiment = 'Gria3'
#experiment = 'Grin2a'
#experiment = 'Hcn4'
#experiment = 'Herc1'
#experiment = 'Nr3c2'
#experiment = 'Sp4'
#experiment = 'Trio'
#experiment = 'Xpo7'
plates, paths, controls, tests = MZU.parse_summary_PPI(summary_path, experiment)

# Set list of video paths
path_list = paths

# Inspect behaviour for video paths (*.avi) in path_list
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
    inspect_folder = output_folder + '/inspect'
    controls_inspect_folder = inspect_folder + '/controls'
    tests_inspect_folder = inspect_folder + '/tests'

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

    # Inspect control responses
    control_paths = glob.glob(controls_inspect_folder+'/*.avi')
    for control_path in control_paths:
        name = os.path.basename(control_path)[:-4]
        well_number = int(name.split('_')[1])
        fish = plate.wells[well_number-1]

        # Review inspection videos and build manual classification table

#FIN
