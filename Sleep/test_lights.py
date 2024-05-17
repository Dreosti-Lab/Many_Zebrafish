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
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from io import StringIO

# Import local modules
import MZ_utilities as MZU

# Reload modules
import importlib
importlib.reload(MZU)
#----------------------------------------------------------
zip_path = "/run/media/kampff/Crucial X9/Zebrafish/Sleep/220919_10_11_gria3xpo7/220919_10_11_gria3xpo7_rawoutput.zip"
zip_path = "/run/media/kampff/Crucial X9/Zebrafish/Sleep/220815_14_15_Gria3Trio/220815_14_15_Gria3Trio_rawoutput.zip"

archive = zipfile.ZipFile(zip_path, 'r')
for zf in archive.filelist:
    xls_data = StringIO(archive.read(zf.filename).decode('ascii'))
    data = pd.read_csv(xls_data, sep='\t')
    lights = np.where(data.type == 109)
    print(lights)

#FIN