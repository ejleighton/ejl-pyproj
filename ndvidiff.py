import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import rasterio as rio
from mpl_toolkits.axes_grid1 import make_axes_locatable  # for arranging multiple plots, i.e. colorbar axes
from cartopy.feature import ShapelyFeature
from rasterio.windows import from_bounds  # thanks https://gis.stackexchange.com/a/336903 for pointing this function out
from myconfig import *  # imports variables from configuration file (per https://stackoverflow.com/a/924866)


# --------------------------------[ FUNCTIONS ]--------------------------------------

def get_outline(crs_src, poly):
    """
    Returns polygon feature with CRS from a source raster.

        Parameters:
            crs_src (str): path and filename for the CRS source raster
            poly: path and filename for the shapefile

        Returns:
            output: polygon feature set to source CRS
    """
    with rio.open(crs_src) as src:
        output = gpd.read_file(poly).to_crs(src.crs)
    return output


def band_clip(filepath):
    """
    Returns a windowed raster from the given path. Assumes a single band.

        Parameters:
            filepath (str): the path and filename for the input raster

        Returns:
            output (array): raster dataset windowed to the polygon boundaries
    """
    with rio.open(filepath) as src:
        output = src.read(window=from_bounds(xmin, ymin, xmax, ymax, src.transform))
    return output


def calc_ndvi(nir, red):
    """
    Returns NDVI from two integer arrays

        Parameters:
            nir (array): near infra-red band raster as an integer array
            red (array): red band raster as an integer array

        Returns:
            ndvi (array): NDVI calculated raster as a floating point array
    """
    nir = nir.astype('float32')
    red = red.astype('float32')
    ndvi = np.where(
        (nir+red) == 0.,
        0,
        (nir - red) / (nir + red)
    )
    return ndvi


# --------------------------------[ DATASETS ]--------------------------------------

# load the polygon and set CRS to same as the Landsat data
outline = get_outline(newRed, shapefile)

# take image bounds from polygon
xmin, ymin, xmax, ymax = outline.total_bounds

# load and clip Landsat data
img1red = band_clip(newRed)
img1nir = band_clip(newNIR)
img2red = band_clip(oldRed)
img2nir = band_clip(oldNIR)


# --------------------------------[ BAND MATHS ]--------------------------------------

ndvi1 = calc_ndvi(img1nir, img1red)
ndvi2 = calc_ndvi(img2nir, img2red)
ndvidiff = (ndvi1-ndvi2)


# --------------------------------[ PLOTTING ]--------------------------------------

# create figure and axes
myCRS = ccrs.Mercator()
fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection=myCRS)
ax.set_extent([xmin, xmax, ymin, ymax], crs=myCRS)

# display raster
im = ax.imshow(ndvidiff[0], cmap='PiYG', vmin=-1, vmax=1, transform=myCRS, extent=[xmin, xmax, ymin, ymax])

# display poly
outline_disp = ShapelyFeature(outline['geometry'], myCRS, edgecolor='r', facecolor='none', linewidth=3.0)
ax.add_feature(outline_disp)

# plot gridlines with labels on top and left
gridlines = ax.gridlines(draw_labels=True)
gridlines.right_labels = False
gridlines.bottom_labels = False

# create axes for colorbar plot
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)

plt.colorbar(im, cax)

# show the plot
plt.show()

# save the plot
# fig.savefig('test.png', dpi=300, bbox_inches='tight')
