# -*- coding: utf-8 -*-
"""
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
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import local modules
import MZ_utilities as MZU

# Reload modules
import importlib
importlib.reload(MZU)

#----------------------------------------------------------
zip_path = base_path + "/Test/grin2agria3/230821_14_grin2agria3_rawoutput.zip"
phc_path = base_path + "/Test/grin2agria3/20230821-130604.phc"
output_base_path = phc_path[:-4]

# Get start time
start_datetime = MZU.get_start_date_time(phc_path)

# Load from file
frametimes_A = np.loadtxt(output_base_path + "_frametimes_A.csv")
frametimes_B = np.loadtxt(output_base_path + "_frametimes_B.csv")

# Compute sunset and sunrise frames
sunsets_A, sunrises_A = MZU.get_sunset_sunrise_frames(start_datetime, frametimes_A)
sunsets_B, sunrises_B = MZU.get_sunset_sunrise_frames(start_datetime, frametimes_B)
print(sunsets_A)
print(sunsets_B)

sunsets_A = [889994, 3049911, 5209874]
sunrises_A = [1789952, 3949903, 6109826]

# Load processed bouts and lights
fish_folder_A = base_path + "/Sleep/230821_14_grin2agria3/Movie"

# Set range
range = 5000

# Load light intensity
intensity_path = fish_folder_A + "/Box1/analysis/figures/intensity.csv"
intensity = np.genfromtxt(intensity_path)

plt.figure()
# Find intensity around sunsets
for i, sunsets in enumerate(sunsets_A):
    plt.subplot(3,1,i+1)
    min_val = sunsets - range
    max_val = sunsets + range
    response = intensity[min_val:max_val]
    plt.axvline(x=0, color='red', linestyle='-', linewidth=1)
    plt.plot(np.arange(-range,range,1), response, alpha=0.5)
plt.savefig("Sleep/_tmp/light_intensity.png")
plt.close()

# Analyse all control bouts
controls_folder = fish_folder_A + "/Box1/analysis/responses/controls"
tests_folder = fish_folder_A + "/Box1/analysis/responses/tests"
control_paths = glob.glob(controls_folder+'/*.npz')
tests_paths = glob.glob(tests_folder+'/*.npz')

plt.figure()
# Find bouts around sunsets
for i, sunsets in enumerate(sunsets_A):
    plt.subplot(3,1,i+1)
    response = np.zeros((2*range)+1)
    for response_path in control_paths+tests_paths:
        name = os.path.basename(response_path)[:-4]
        well_number = int(name.split('_')[1])
        data = np.load(response_path)
        bouts = data['bouts']
        min_val = sunsets - range
        max_val = sunsets + range
        sunset_bouts = bouts[(bouts[:, 0] >= min_val) & (bouts[:, 0] <= max_val)]
        for b in sunset_bouts:
            start_offset = int(b[0] - sunsets + range)
            stop_offset = int(b[2] - sunsets + range)
            response[start_offset:stop_offset] = response[start_offset:stop_offset] + 1
    plt.axvline(x=0, color='red', linestyle='-', linewidth=1)
    plt.plot(np.arange(-range,range+1,1), response, alpha=0.5)
plt.savefig("Sleep/_tmp/light_bouts.png")
plt.close()
#FIN