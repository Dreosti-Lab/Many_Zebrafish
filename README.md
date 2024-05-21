# Many_Zebrafish
A set of Python analysis scripts for analysing behaviour of ***many*** zebrafish in 96-well plates

## Virtual Environment
It is a really good idea to create a "virtual" environment for python that is dedicated to running the code in this repository. Instructions for Mac/Linux follow.

- Create a "_tmp" folder in the Repo root
```bash
mkdir _tmp
```
- Navigate to "_tmp" and create a python virtual environment
```bash
cd _tmp
python3 -m venv MZPY
```
- Activate the virtual environment
```bash
# Mac/Linux
source MZPY/bin/activate

# Windows
./MZPY/Scripts/activate.ps1

# Note: you may need to allow scripts to run on PowerShell
# - Open PowerShell as administrator
#    (right-click and select "Run as Adminstrator") from the Windows start menu
# PS> Set-ExecutionPolicy RemoteSigned
```
- You should now see ***(MZPY)*** activated in the terminal
- Now use *pip* to install the required python packages (see below)

## Python Package Requirements
- numpy
- scipy
- matplotlib
- opencv-python
- scikit-image
- python-dotenv
- openpyxl
- seaborn

For machine learning classifier:
- torch
- torchvision
- torchsummary

```bash
# Ensure your virtual environment is (active)
pip install numpy matplotlib opencv-python scikit-image python-dotenv openpyxl seaborn
pip install torch torchvision torchsummary
```

## Environment File
You must create a ".env" file in the root directory of the Repository with equivalent content to the following, obviously with **paths** that make sense for your computer:
```txt
LIBS_PATH="/home/kampff/Repos/Dreosti-Lab/Many_Zebrafish/libs"
BASE_PATH="/run/media/kampff/Data/Zebrafish/"
```

## Analysis Steps
All the results of analysis are stored in an "analysis" folder located within the *output_folder* (which is typically the folder with the experiment movie).

0. **Step0_check**: Generates an initial "background" and accumulated "motion" (i.e. difference) image
   - The motion should eb accumulated for longer than the initial background is generated, as this difference image is used to detect the 96 ROIs
   - Confirm that the background and difference images are sensible before proceeding
1. **Step1_rois**: Automatically detects the ROIs for each of the 96-wells
   - Confirm that the ROIs are accurately detected and localised before proceeding
2. **Step2_measure**: Tracks each fish in each ROI, dynamically updates background
   - Set the "background update rate" based on movie frame rate (4 secs between updates is good)
   - If "validate" is True, then an overview of the tracking will be saved every 1000 frames
3. **Step3_summarise**: Generate intermediate tracking data and summary figures

