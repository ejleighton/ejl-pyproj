# This file is for the configuration of paths and filenames to be passed to the main script.
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

maptitle = 'Difference in NDVI in the Ukraine CEZ from May 1986 to May 1991'
newRed = 'data_files\\CEZ\\LT05_L1TP_182024_19910513_20180225_01_T1_B3.TIF'  # image 1 visible red
newNIR = 'data_files\\CEZ\\LT05_L1TP_182024_19910513_20180225_01_T1_B4.TIF'  # image 1 nir
oldRed = 'data_files\\CEZ\\LT05_L1TP_182024_19860531_20170217_01_T1_B3.TIF'  # image 2 visible red
oldNIR = 'data_files\\CEZ\\LT05_L1TP_182024_19860531_20170217_01_T1_B4.TIF'  # image 2 nir
shapefile = 'data_files\\CEZ\\ExclusionZone.shp'  # shapefile of study area
spatialref = 'epsg:5577'  # espg for Ukraine 2000 3 Degree GK CM 21E
posthresh, negthresh = 0.3, -0.3
