"""
This file is for the configuration of paths and filenames to be passed to the main script.

All variables must be configured.

Variables:
    newRed: path and filename for the red band of the more recent image
    newNIR: path and filename for the NIR band of the more recent image
    oldRed: path and filename for the red band of the older image
    oldNIR: path and filename for the NIR band of the older image
    shapefile: path and filename for the shapefile containing the study area boundary
"""

newRed = 'data_files\\LT05_L1TP_182024_19910513_20180225_01_T1_B3.TIF'  # image 1 visible red
newNIR = 'data_files\\LT05_L1TP_182024_19910513_20180225_01_T1_B4.TIF'  # image 1 nir
oldRed = 'data_files\\LT05_L1TP_182024_19860531_20170217_01_T1_B3.TIF'  # image 2 visible red
oldNIR = 'data_files\\LT05_L1TP_182024_19860531_20170217_01_T1_B4.TIF'  # image 2 nir
shapefile = 'data_files\\ExclusionZone.shp'  # shapefile of study area