# -*- coding: utf-8 -*-
"""
Quickly check the result of a 96-well experiment

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
import libs.MZ_video as MZV

# Reload modules
import importlib
importlib.reload(MZV)

# Input path
input_path = base_path + r'/PPI_Behaviour/240123_nr3c2_PPI_Exp000.avi'
#input_path = base_path + r'/PPI_Behaviour/231218_herc1_PPI_stim_Exp00.avi'

# Output folder
output_folder = base_path + r'/PPI_Behaviour/output'

# Process summary images
MZV.process_video_summary_images(input_path, output_folder)

#FIN
