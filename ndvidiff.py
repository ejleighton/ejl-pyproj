import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import rasterio as rio
import rasterio.mask
from cartopy.feature import ShapelyFeature
from rasterio.windows import from_bounds  # thanks https://gis.stackexchange.com/a/336903 for pointing this function out!

# --------------------------------[ FUNCTIONS ]--------------------------------------

def band_clip(filepath):
    '''
    Returns a windowed raster from the given path. Assumes a single band.

        Parameters:
            filepath (str): the path and filename for the input raster

        Returns:
            raster dataset windowed to the polygon boundaries
    '''
    with rio.open(filepath) as img:
        output = img.read(window=from_bounds(xmin,ymin,xmax,ymax,img.transform))
    return output

def calc_ndvi(nir,red):
    '''
    Returns NDVI from two integer arrays

        Parameters:
            nir (array): near infra-red band raster as an integer array
            red (array): red band raster as an integer array

        Returns:
            ndvi (array): NDVI calculated raster as a floating point array
    '''
    nir = nir.astype('float32')
    red = red.astype('float32')
    ndvi = np.where(
        (nir+red) == 0.,
        0,
        (nir - red) / (nir + red)
    )  # note this operation will trigger divide by zero errors, but the function will complete
    return ndvi

# --------------------------------[ DATASETS ]--------------------------------------

# Get CRS from first Landsat image
with rio.open('data_files\\LT05_L1TP_182024_19910513_20180225_01_T1_B3.TIF') as img:
    rastcrs = img.crs  # save the Landsat image CRS as a variable

# load the polygon and set CRS to same as the Landsat data
outline = gpd.read_file('data_files\\ExclusionZone.shp').to_crs(rastcrs)

# taking image bounds from polygon
xmin, ymin, xmax, ymax = outline.total_bounds


# load Landsat data
img1red = band_clip('data_files\\LT05_L1TP_182024_19910513_20180225_01_T1_B3.TIF')  # image 1 visible red
img1nir = band_clip('data_files\\LT05_L1TP_182024_19910513_20180225_01_T1_B4.TIF')  # image 1 nir
img2red = band_clip('data_files\\LT05_L1TP_182024_19860531_20170217_01_T1_B3.TIF')  # image 2 visible red
img2nir = band_clip('data_files\\LT05_L1TP_182024_19860531_20170217_01_T1_B4.TIF')  # image 2 nir


# --------------------------------[ BAND MATHS ]--------------------------------------

ndvi1 = calc_ndvi(img1nir,img1red)
ndvi2 = calc_ndvi(img2nir,img2red)
ndvidiff = (ndvi1-ndvi2)

# --------------------------------[ PLOTTING ]--------------------------------------

# create figure and axes
myCRS = ccrs.UTM(35)
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=myCRS)
ax.set_extent([xmin, xmax, ymin, ymax], crs=myCRS)

# display raster
ax.imshow(ndvidiff[0], cmap='viridis', vmin=-1, vmax=1, transform=myCRS, extent=[xmin, xmax, ymin, ymax])

# display poly
outline_disp = ShapelyFeature(outline['geometry'], myCRS, edgecolor='r', facecolor='none', linewidth=3.0)
ax.add_feature(outline_disp)

# plot gridlines with labels on top and left
gridlines = ax.gridlines(draw_labels=True)
gridlines.right_labels = False
gridlines.bottom_labels = False

# show the plot
plt.show()

# save the plot
fig.savefig('test.png', dpi=300, bbox_inches='tight')