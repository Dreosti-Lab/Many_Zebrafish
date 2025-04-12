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
import pandas as pd
import zipfile
from io import StringIO
import re
from datetime import datetime, timedelta

# Import local modules
import MZ_utilities as MZU

# Reload modules
import importlib
importlib.reload(MZU)
#----------------------------------------------------------
zip_path = base_path + "/220919_10_11_gria3xpo7_rawoutput.zip"
phc_path = base_path + "/20220919-190935.phc"

# Extract experiment start time
def get_start_date_time(phc_path):
    # Read first line of PHC file
    with open(phc_path, 'r', encoding='utf-8') as f:
        line = f.readline()

    # Extract RunDate and RunTime using regex
    match = re.search(r'RunDate="([\d/]+)".*?RunTime="([\d:]+ [APM]+)"', line)
    if match:
        date_str = match.group(1)  # e.g., '2022/09/19'
        time_str = match.group(2)  # e.g., '7:09:55 PM'
        
        # Combine and parse into datetime
        combined_str = f"{date_str} {time_str}"
        dt = datetime.strptime(combined_str, "%Y/%m/%d %I:%M:%S %p")
        
        # Convert to UNIX timestamp
        timestamp = dt.timestamp()
        return timestamp
    else:
        raise ValueError("RunDate and RunTime not found in the first line.")

# Initialize sets to store unique timestamps per category
timestamps_A = set()
timestamps_B = set()

# Open the archive
archive = zipfile.ZipFile(zip_path, 'r')
xls_files = [zf for zf in archive.filelist if zf.filename.lower().endswith(('.xls', '.xlsx'))]

# Loop with progress bar (or manual counter)
for i, zf in enumerate(xls_files):
    try:
        raw = archive.read(zf.filename).decode('ascii')
        df = pd.read_csv(StringIO(raw), sep='\t')
    except Exception as e:
        print(f"Error reading {zf.filename}: {e}")
        continue

    # Filter rows with numeric locations and timestamps
    df = df[df['location'].str.startswith('C') & df['abstime'].apply(lambda x: str(x).isdigit())]

    # Extract location numbers
    df['loc_num'] = df['location'].str[1:].astype(int)
    df['ts'] = df['abstime'].astype(int)

    # Apply masks for location ranges
    mask_A = df['loc_num'].between(1, 96)
    mask_B = df['loc_num'].between(97, 192)

    # Add timestamps to corresponding sets
    timestamps_A.update(df.loc[mask_A, 'ts'])
    timestamps_B.update(df.loc[mask_B, 'ts'])

    # Report
    print(f"Processed {i}/{len(xls_files)}: {zf.filename}")

# Convert to sorted lists
sorted_A = sorted(timestamps_A)
sorted_B = sorted(timestamps_B)

# Optional: print counts
print(f"Timestamps A: {len(sorted_A)} unique values")
print(f"Timestamps B: {len(sorted_B)} unique values")

# Optional: preview
print("First few from A:", sorted_A[:5])
print("First few from B:", sorted_B[:5])

# Convert to seconds
timestamps_A_sec = np.array(sorted_A) / 1e6
timestamps_B_sec = np.array(sorted_B) / 1e6

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

# Assume:
# sorted_A is a list of timestamps in microseconds
# start_datetime is the datetime object representing the first timestamp's absolute time

# Report start time
timestamp = get_start_date_time(phc_path)
start_datetime = datetime.fromtimestamp(timestamp)
print(start_datetime)

# Convert sorted list to NumPy array for fast search
timestamps_A_us = np.array(sorted_A) - sorted_A[0]
timestamps_B_us = np.array(sorted_B) - sorted_B[0]

MICROSECONDS_IN_DAY = 24 * 60 * 60 * 1_000_000
MICROSECONDS_IN_14_HOURS = 14 * 60 * 60 * 1_000_000

# 1. Compute first sunset (11 PM on the same day, or next day if already past)
first_sunset_dt = start_datetime.replace(hour=23, minute=0, second=0, microsecond=0)
if start_datetime >= first_sunset_dt:
    print("Experiment Flaw: started after 11 pm")
    first_sunset_dt += timedelta(days=1)

# Compute microsecond offset from start to first sunset
first_sunset_us = int((first_sunset_dt - start_datetime).total_seconds() * 1e6)

# Initialize lists
sunset_offsets = []
sunrise_offsets = []

# Max range of data
max_A_us = timestamps_A_us[-1]
max_B_us = timestamps_B_us[-1]
max_us = min(max_A_us, max_B_us)

# 2. Loop to find all sunset/sunrise times (A)
current_sunset_us = first_sunset_us
while current_sunset_us <= max_us:
    sunset_offsets.append(current_sunset_us)
    sunrise_offsets.append(current_sunset_us - MICROSECONDS_IN_14_HOURS)  # sunrise = 14 hrs before sunset
    current_sunset_us += MICROSECONDS_IN_DAY

# 3. Use searchsorted to get indices
sunset_indices_A = np.searchsorted(timestamps_A_us, sunset_offsets, side='right')
sunrise_indices_A = np.searchsorted(timestamps_A_us, sunrise_offsets, side='right')

sunset_indices_B = np.searchsorted(timestamps_B_us, sunset_offsets, side='right')
sunrise_indices_B = np.searchsorted(timestamps_B_us, sunrise_offsets, side='right')

#FIN