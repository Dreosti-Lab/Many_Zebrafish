# -*- coding: utf-8 -*-
"""
Many_Zebrafish: Utility Library

@author: kamnpff (Adam Kampff)
"""
# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import cv2

# Utilities for analysing 96-well plate experiments

# Extract PPI stimulus from LED intensity time series
def extract_ppi_stimuli(led_intensity):
    baseline = np.median(led_intensity)
    signal = led_intensity - baseline
    threshold = np.max(signal) / 4
    pulses = []
    last_peak = 0
    for i, s in enumerate(signal):
        if s > threshold:
            if (i - last_peak) > 3:
                pulses.append(i)
                last_peak = i
    single_pulses = []
    paired_pulses = []
    p = 0
    while p < len(pulses):
        if (p+1) == len(pulses):
            single_pulses.append(pulses[p])
            break
        if (pulses[p+1] - pulses[p]) > 1000:
            single_pulses.append(pulses[p])
            p = p + 1
        else:
            paired_pulses.append((pulses[p], pulses[p+1]))
            p = p + 2
            
    return single_pulses, paired_pulses

#FIN