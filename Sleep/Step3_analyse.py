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
import matplotlib.pyplot as plt

# Import modules
import MZ_plate as MZP
import MZ_video as MZV
import MZ_utilities as MZU

# Reload modules
import importlib
importlib.reload(MZP)
importlib.reload(MZV)
importlib.reload(MZU)

#----------------------------------------------------------

# Load list of video paths
path_list_path = base_path + "/path_list.txt"
path_list = MZU.load_path_list(path_list_path)

# Analyse behaviour for video paths (*.avi) in path_list
for path in path_list:
    # Create Paths
    video_path = base_path + path
    output_folder = os.path.dirname(video_path) + '/analysis'
    plate_folder = output_folder + '/plate'
    figures_folder = os.path.dirname(video_path) + '/analysis/figures'
    fish_figures_folder = os.path.dirname(video_path) + '/analysis/figures/fish'
    
    # Create figures folder
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)   

    # Create fish figures folder
    if not os.path.exists(fish_figures_folder):
        os.makedirs(fish_figures_folder)   

    # Create plate structure
    name = path.split('/')[-1][:-4]
    plate = MZP.Plate(name)

    # Load plate in chunks
    plate_paths = sorted(glob.glob(plate_folder + '/*.npz'), key=os.path.getmtime)
    count  = 0
    intensity = np.empty(0, dtype=np.float32)
    for plate_path in plate_paths:
        frame_range = plate_path[:-4].split('_')[-2:]
        start_frame = int(frame_range[0])
        end_frame = int(frame_range[1])
        print(f'Loading plate data chunk...{start_frame} to {end_frame}')
        plate.load(output_folder, start_frame, end_frame)
        if count >= 1:
            print(np.sum(plate.wells[11].stack[2][:] - previous_debug))
        previous_debug = plate.wells[11].stack[2][:]
        count = count + 1
        print(plate.intensity[-1])
        intensity = np.hstack((intensity, plate.intensity))
        plate.clear()

    # ---- Need and append plate function ?? ----

    # Load intensity
    num_frames = len(intensity)
    print(num_frames)

    # Plot intensity
    fig = plt.figure(figsize=(10, 4))
    plt.title('Background Intensity Detection')
    plt.plot(intensity)
    plt.savefig(figures_folder + '/intensity.png', dpi=180)
    plt.cla()
    plt.close()
    
    # Analyse
    for i, fish in enumerate(plate.wells):
        figure_path = fish_figures_folder + f'/{(i+1):02d}_fish.png'
        x = fish.x
        y = fish.y
        area = fish.area
        heading = fish.heading
        motion = fish.motion

        fig = plt.figure(figsize=(10, 8))
        plt.subplot(2,2,1)
        plt.title('Motion')
        plt.plot(motion)
        plt.ylim(0, 10000)
        plt.subplot(2,2,2)
        plt.title('Tracking')
        plt.plot(x,y,'.', markersize=3, color=[0,0,0,0.01])
        plt.xlim(fish.ul[0], fish.lr[0])
        plt.ylim(fish.ul[1], fish.lr[1])
        plt.subplot(2,2,3)
        plt.title('Area')
        plt.plot(area, 'm.', markersize=2, alpha=0.25)
        plt.ylim(0, 400)
        plt.subplot(2,2,4)
        plt.title('Heading')
        plt.plot(heading, 'g.', markersize=2, alpha=0.25)

        # Save
        plt.savefig(figure_path, dpi=180)
        print(i)
        plt.cla()
        plt.close()

#FIN
