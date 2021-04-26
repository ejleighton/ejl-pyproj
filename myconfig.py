"""
This file is for the configuration of paths and filenames to be passed to the main script.

All variables must be configured.

Variables:
    maptitle: text to appear above the final map figure
    newRed: path and filename for the red band of the more recent image
    newNIR: path and filename for the NIR band of the more recent image
    oldRed: path and filename for the red band of the older image
    oldNIR: path and filename for the NIR band of the older image
    shapefile: path and filename for the shapefile containing the study area boundary
"""

maptitle = 'Difference in NDVI in the Mt. St. Helens National Volcano Monument from Sept 1986 to July 1995'
newRed = 'data_files\\MSH\\LT05_L1TP_046028_19950718_20160926_01_T1_B3.TIF'  # image 1 visible red
newNIR = 'data_files\\MSH\\LT05_L1TP_046028_19950718_20160926_01_T1_B4.TIF'  # image 1 nir
oldRed = 'data_files\\MSH\\LT05_L1TP_046028_19860911_20161003_01_T1_B3.TIF'  # image 2 visible red
oldNIR = 'data_files\\MSH\\LT05_L1TP_046028_19860911_20161003_01_T1_B4.TIF'  # image 2 nir
shapefile = 'data_files\\MSH\\MSH_NVM.shp'  # shapefile of study area
