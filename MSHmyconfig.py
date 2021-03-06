# This file is for the configuration of variables to be passed to the main script.
#
# All variables must be configured.
#
# Variables:
#    maptitle: text to appear above the final map figure
#    newRed: path and filename for the red band of the more recent image
#    newNIR: path and filename for the NIR band of the more recent image
#    oldRed: path and filename for the red band of the older image
#    oldNIR: path and filename for the NIR band of the older image
#    shapefile: path and filename for the shapefile containing the study area boundary
#    spatialref: spatial reference for analysis in format 'epsg:#####'
#    posthresh, negthresh: positive and negative thresholds to be applied to NDVI difference image to
#                          determine significant positive or negative change
#    savepath: path to save output images

maptitle = 'NDVI Difference\nMt. St. Helens National Volcano Monument\nSept 1986 to July 1995'
newRed = 'data_files\\MSH\\LT05_L1TP_046028_19950718_20160926_01_T1_B3.TIF'  # image 1 visible red
newNIR = 'data_files\\MSH\\LT05_L1TP_046028_19950718_20160926_01_T1_B4.TIF'  # image 1 nir
oldRed = 'data_files\\MSH\\LT05_L1TP_046028_19860911_20161003_01_T1_B3.TIF'  # image 2 visible red
oldNIR = 'data_files\\MSH\\LT05_L1TP_046028_19860911_20161003_01_T1_B4.TIF'  # image 2 nir
shapefile = 'data_files\\MSH\\MSH_NVM.shp'  # shapefile of study area
spatialref = 'epsg:7582'  # espg for NAD83(2011) / WISCRS Washington (m)
posthresh, negthresh = 0.0, -0.0
savepath = 'MSH_output' 
