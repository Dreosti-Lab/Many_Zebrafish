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
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import local modules
import MZ_utilities as MZU

# Reload modules
import importlib
importlib.reload(MZU)

#----------------------------------------------------------
#zip_path = base_path + "/grin2agria3/230821_14_grin2agria3_rawoutput.zip"
#phc_path = base_path + "/grin2agria3/20230821-130604.phc"
#output_base_path = phc_path[:-4]

zip_path = base_path + "/Sleep/220815_14_15_Gria3Trio/220815_14_15_Gria3Trio_rawoutput.zip"
phc_path = base_path + "/Sleep/220815_14_15_Gria3Trio/220815_14_15_Gria3Trio.phc"
output_base_path = phc_path[:-4]

# Get start time
start_datetime = MZU.get_start_date_time(phc_path)

# Get frame times
frametimes_A, frametimes_B = MZU.get_ordered_frame_times(zip_path)

# Store frame times
with open(output_base_path + "_frametimes_A.csv", 'w') as file:
    for item in frametimes_A:
        file.write(f"{item}\n")
with open(output_base_path + "_frametimes_B.csv", 'w') as file:
    for item in frametimes_B:
        file.write(f"{item}\n")

# Print frame counts
print(f"Timestamps A: {len(frametimes_A)} unique values")
print(f"Timestamps B: {len(frametimes_B)} unique values")

# Convert to seconds
timestamps_A_sec = np.array(frametimes_A) / 1e6
timestamps_B_sec = np.array(frametimes_B) / 1e6

# Compute time differences between frames
intervals_A = np.diff(timestamps_A_sec)
intervals_B = np.diff(timestamps_B_sec)

# Compute average frame interval and frame rate
if len(intervals_A) > 0:
    avg_interval_A = np.mean(intervals_A)
    frame_rate_A = 1.0 / avg_interval_A
    print(f"Frame rate for A: {frame_rate_A:.5f} fps")
else:
    print("Not enough data to compute frame rate for A")
if len(intervals_B) > 0:
    avg_interval_B = np.mean(intervals_B)
    frame_rate_B = 1.0 / avg_interval_B
    print(f"Frame rate for B: {frame_rate_B:.5f} fps")
else:
    print("Not enough data to compute frame rate for B")

# Compute sunset and sunrise frames
sunsets_A, sunrises_A = MZU.get_sunset_sunrise_frames(start_datetime, frametimes_A)
sunsets_B, sunrises_B = MZU.get_sunset_sunrise_frames(start_datetime, frametimes_B)

#FIN