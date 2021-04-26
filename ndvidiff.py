import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import rasterio as rio
import rasterio.mask as rmask
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


def getextent(poly):
    """
    Returns variables for extent of image from polygon bounds
    Grows extent by 1%

    Parameters:
        poly: path and filename of shapefile

    Returns:
        minx, miny, maxx, maxy: min and max extent of polygon increased by 1% each direction
    """
    # takes bounds from the polygon
    minx, miny, maxx, maxy = poly.total_bounds
    # calculates 1% of the bounding box
    deltax = ((maxx - minx) / 100)
    deltay = ((maxy - miny) / 100)
    # increases the minimum and maximum extent by the calculated amount
    minx = (minx - deltax)
    miny = (miny - deltay)
    maxx = (maxx + deltax)
    maxy = (maxy + deltay)
    # returns the results to the variables
    return minx, miny, maxx, maxy
    # this is purely for aesthetics preventing the polygon from being right next to the axes edge


def band_clip(filepath, outpath):
    """
    Returns a windowed raster from the given path. Assumes a single band.
    Creates a Geotiff of the resulting image with updated metadata.

        Parameters:
            filepath (str): the path and filename for the input raster
            outpath (str): the path and filename for the output image

        Returns:
            output (array): raster dataset windowed to the polygon boundaries
    """
    with rio.open(filepath) as src:
        win = from_bounds(xmin, ymin, xmax, ymax, src.transform)
        out_img = src.read(window=win)
        out_transform = src.window_transform(win)
        out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform})
    with rio.open(outpath, "w", **out_meta) as dest:
        dest.write(out_img)
    return out_img


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


def ndvidiff(new, old, outpath):
    """
    Returns a difference raster from the two input rasters.
    Creates a GeoTiff of the output

        Parameters:
            new (array): more recent ndvi image as floating point array
            old (array): older ndvi image as floating point array
            outpath (str): the path and filename for the output image

        Returns:
            diff (array): raster created subtracting the older image from the new
    """
    diff = (new - old)
    with rio.open('output\\img1red.tif') as src:
        out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": diff.shape[1],
                     "width": diff.shape[2],
                     "dtype": "float32"})
    with rio.open(outpath, "w", **out_meta) as dest:
        dest.write(diff)
    return diff


# --------------------------------[ DATASETS ]--------------------------------------

# load the polygon and set CRS to same as the Landsat data
outline = get_outline(newRed, shapefile)

# take image bounds from polygon
xmin, ymin, xmax, ymax = getextent(outline)

# load and clip Landsat data, saves as GeoTiff
img1red = band_clip(newRed, 'output\\img1red.tif')
img1nir = band_clip(newNIR, 'output\\img1nir.tif')
img2red = band_clip(oldRed, 'output\\img2red.tif')
img2nir = band_clip(oldNIR, 'output\\img2nir.tif')


# --------------------------------[ BAND MATHS ]--------------------------------------

ndvi1 = calc_ndvi(img1nir, img1red)
ndvi2 = calc_ndvi(img2nir, img2red)
diffimg = ndvidiff(ndvi1, ndvi2, 'output\\ndvidiff.tif')

# create masked copy of the NDVI difference layer
with rio.open('output\\ndvidiff.tif') as img:
    # set any pixel outside the 'outline' polygon to a value of -9999
    masked_diff, mask_transform = rmask.mask(img, outline.geometry, nodata=-9999)
# create a masked array with numpy setting -9999 as a 'bad' or out-of-range value
masked_diff = np.ma.masked_where(masked_diff == -9999, masked_diff)


# --------------------------------[ PLOTTING ]--------------------------------------

# create figure and axes
myCRS = ccrs.UTM(35)
fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection=myCRS)
ax.set_extent([xmin, xmax, ymin, ymax], crs=myCRS)
plt.title('NDVI Difference')

# create colormap setting alpha for masked values
cmap = plt.cm.get_cmap("RdYlBu").copy()
cmap.set_bad('k', alpha=0)

# ax.stock_img()  # displays a (very) low res natural earth background, but slows the plotting significantly

# display raster

im = ax.imshow(masked_diff[0], cmap=cmap, vmin=-0.5, vmax=0.5, transform=myCRS, extent=[xmin, xmax, ymin, ymax])

# display poly
outline_disp = ShapelyFeature(outline['geometry'], myCRS, edgecolor='r', facecolor='none', linewidth=3.0)
ax.add_feature(outline_disp)

# plot gridlines with labels on top and left
gridlines = ax.gridlines(draw_labels=True)
gridlines.right_labels = False
gridlines.bottom_labels = False

# create axes for colorbar plot
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1, axes_class=plt.Axes)

plt.colorbar(im, cax)


# show the plot
plt.show()

# save the plot
# fig.savefig('output\\test.png', dpi=300, bbox_inches='tight')
