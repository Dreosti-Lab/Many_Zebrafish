# -*- coding: utf-8 -*-
"""
Many_Zebrafish: Plate Class

@author: kampff
"""
# Import useful libraries
import numpy as np

# Import local modules
from MZ_fish import Fish

# Plate Class
class Plate:
    def __init__(self, _name):
        self.intensity = []
        self.wells = []
        for i in range(96):
            fish = Fish()
            self.wells.append(fish)
        return

    def init_backgrounds(self, background):
        for fish in self.wells:
            fish.init_background(background)
        return

    def save(self, output_folder, start_frame=0, end_frame=-1):
        if end_frame == -1:
            end_frame = start_frame + len(self.intensity)
        num_frames = end_frame - start_frame
        for i, fish in enumerate(self.wells):
            fish_path = output_folder + f'/fish/{(i+1):02d}_fish.csv'
            fish_behaviour = np.zeros((num_frames,5), dtype=np.float32)
            fish_behaviour[:,0] = np.array(fish.x)
            fish_behaviour[:,1] = np.array(fish.y)
            fish_behaviour[:,2] = np.array(fish.area)
            fish_behaviour[:,3] = np.array(fish.heading)
            fish_behaviour[:,4] = np.array(fish.motion)
            # Savez here! but pack in intensity, call it wells
            np.savetxt(fish_path, fish_behaviour, fmt='%.3f', delimiter=',')
        intensity_path = output_folder + '/intensity.csv'
        intensity_array = np.array(self.intensity)
        np.savetxt(intensity_path, intensity_array, fmt='%d')
        return

# FIN


# Load ROIs
def load_rois(roi_path, plate):
    roi_data = np.genfromtxt(roi_path, delimiter=',')
    for i in range(96):
        roi_line = roi_data[i]
        ul = (roi_line[0],roi_line[1])
        lr = (roi_line[2],roi_line[3])
        plate[i].set_roi(ul, lr)
    return plate
