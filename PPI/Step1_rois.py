# -*- coding: utf-8 -*-
"""
Automatically extract ROIs from a 96-well experiment

@author: kamnpff (Adam Kampff)
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'/home/kampff/Repos/Dreosti-Lab/Many_Zebrafish/libs'
#-----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Set "Base Path" for this analysis session
base_path = r'/run/media/kampff/Data/Zebrafish/'
#base_path = r'\\128.40.155.187\data\D R E O S T I   L A B'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Import local modules
import MZ_fish as MZF
import MZ_roi as MZR

# Reload modules
import importlib
importlib.reload(MZF)
importlib.reload(MZR)

# Input path
input_path = base_path + r'/PPI_Behaviour/output/difference.png'

# Output folder
output_folder = base_path + r'/PPI_Behaviour/output'

# Create plate structure
plate = MZF.create_plate()

# Process summary images
plate = MZR.find_rois(input_path, plate, output_folder)

#FIN
