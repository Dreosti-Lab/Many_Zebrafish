# Many_Zebrafish
A set of Python analysis scripts for analysing behaviour of ***many*** zebrafish in 96-well plates

## Python package requirements
- numpy
- pandas
- pyarrow
- matplotlib
- opencv-python
- scikit-image
- python-dotenv

## Environment
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
3. **Step3_analyse**: Preliminary analysis scripts working with the intermediate tracking data

