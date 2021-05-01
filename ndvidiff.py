import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import rasterio as rio
import rasterio.mask as rmask
from pathlib import Path
from skimage.filters import threshold_otsu
from rasterio.features import shapes
from rasterio.windows import from_bounds  # thanks https://gis.stackexchange.com/a/336903 for pointing this function out
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable  # for arranging multiple plots, i.e. colorbar axes
from cartopy.feature import ShapelyFeature
from myconfig import *  # imports variables from configuration file (per https://stackoverflow.com/a/924866)


# --------------------------------[ FUNCTIONS ]--------------------------------------

def get_outline(crs_src, poly):
    """Returns polygon feature with CRS from a source raster.

    :param crs_src: (str) path and filename for the CRS source raster
    :param poly: (str) path and filename for the shapefile
    :return: polygon feature set to source CRS
    """
    with rio.open(crs_src) as src:
        output = gpd.read_file(poly).to_crs(src.crs)
    return output


def getextent(poly):
    """Returns variables for extent of image from polygon bounds
    Grows extent by 1%

    :param poly: (str) path and filename of shapefile
    :return: minx, miny, maxx, maxy: min and max extent of polygon increased by 1% each direction
        deltax, deltay: width and height of the source polygon bounding box
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
    return minx, miny, maxx, maxy, deltax, deltay
    # this is purely for aesthetics preventing the polygon from being right next to the axes edge


def band_clip(filepath, outpath):
    """Returns a windowed raster from the given path. Assumes a single band.
    Creates a Geotiff of the resulting image with updated metadata.

    :param filepath: (str) the path and filename for the input raster
    :param outpath: (str) the path and filename for the output image
    :return: output: (array) raster dataset windowed to the polygon boundaries
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
    """Returns NDVI from two integer arrays

    :param nir: (array) near infra-red band raster as an integer array
    :param red: (array) red band raster as an integer array
    :return: ndvi (array): NDVI calculated raster as a floating point array
    """
    nir = nir.astype('float32')
    red = red.astype('float32')
    ndvi = np.where(
        (nir + red) == 0.,
        0,
        (nir - red) / (nir + red)
    )
    return ndvi


def ndvidiff(new, old, outpath):
    """Returns a difference raster from the two input rasters.
    Creates a GeoTiff of the output

    :param new: (array) more recent ndvi image as floating point array
    :param old: (array) older ndvi image as floating point array
    :param outpath: (str): the path and filename for the output image
    :return: (array): raster created subtracting the older image from the new
    """
    diff = (new - old)
    with rio.open('{}\\img1red.tif'.format(savepath)) as src:
        out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": diff.shape[1],
                     "width": diff.shape[2],
                     "dtype": "float32"})
    with rio.open(outpath, "w", **out_meta) as dest:
        dest.write(diff)
    return diff


def get_threshold(srcimg):
    """Returns thresholds for the NDVI differencing image determined by Otsu's method

    :param srcimg: (str) path and filename of the source image
    :return: upper and lower thresholds signifying significant increase or decrease in NDVI
    """
    with rio.open(srcimg) as src:
        pos_array = src.read()
        neg_array = src.read()
    pos_array[np.where(pos_array < 0.0)] = 0.0
    neg_array[np.where((neg_array > 0.0) | (neg_array == -9999))] = 0.0
    upper = threshold_otsu(pos_array)
    lower = threshold_otsu(neg_array)
    print("Thresholds determined by Otsu's method: {:.4f}, {:.4f}".format(upper, lower))
    return upper, lower


# https://gis.stackexchange.com/a/177675
def classify_change(srcimg, pos_thold, neg_thold, outpath):
    """Returns classified image using supplied thresholds

    :param srcimg: (str) path to raster to be classified
    :param pos_thold: (float) value above which pixels will be classed as positive change
    :param neg_thold: (float) value below which pixels will be classed as negative change
    :param outpath: (str) path and filename of output GeoTiff
    :return: classified (array): raster reclassified
    """
    with rio.open(srcimg) as src:
        array = src.read()
        out_meta = src.meta
    array[np.where(array >= pos_thold)] = 1
    array[np.where((array <= neg_thold) & (array > -10))] = 2
    array[np.where((array < pos_thold) & (array > neg_thold))] = 0
    with rio.open(outpath, "w", **out_meta) as dest:
        dest.write(array)
    return array


# https://gis.stackexchange.com/a/187883
def getpolygons(srcimg):
    """Returns polygons based on unique values in source raster

    :param srcimg: (str)  path to raster to be polygonised
    :return: GeoDataFrame of polygons created from the raster
    """
    with rio.Env():
        with rio.open(srcimg) as src:
            image = src.read()
            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v)
                in enumerate(
                    shapes(image, transform=src.transform)))
    geoms = list(results)
    gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms)
    gpd_polygonized_raster.set_crs(crs=src.crs, inplace=True)
    return gpd_polygonized_raster


def getarea(poly):
    """Returns area of polygons in km². Assumes units of source are metres.

    :param poly: shapes to be measured
    :return: area (float): area of each polygon in the source in km²
    """
    geom = poly.to_crs(spatialref)
    area = (geom.area / 1000000)
    return area


# --------------------------------[ DATASETS ]--------------------------------------

# Create an output directory if it doesn't exist
Path(savepath).mkdir(parents=True, exist_ok=True)

# load the polygon and set CRS to same as the Landsat data
outline = get_outline(newRed, shapefile)

# take image bounds from polygon
xmin, ymin, xmax, ymax, o_width, o_height = getextent(outline)

# load and clip Landsat data, saves as GeoTiff
img1red = band_clip(newRed, '{}\\img1red.tif'.format(savepath))
img1nir = band_clip(newNIR, '{}\\img1nir.tif'.format(savepath))
img2red = band_clip(oldRed, '{}\\img2red.tif'.format(savepath))
img2nir = band_clip(oldNIR, '{}\\img2nir.tif'.format(savepath))

outlinearea = getarea(outline).sum()
print('Study area size: {:.2f} km²'.format(outlinearea))
print('Study area size (UTM): {:.2f} km²'.format(outline.area[0] / 1000000))

# --------------------------------[ ANALYSIS ]--------------------------------------

ndvi1 = calc_ndvi(img1nir, img1red)
ndvi2 = calc_ndvi(img2nir, img2red)
diffimg = ndvidiff(ndvi1, ndvi2, '{}\\ndvidiff.tif'.format(savepath))

# create masked copy of the NDVI difference layer
with rio.open('{}\\ndvidiff.tif'.format(savepath)) as img:
    # set any pixel outside the 'outline' polygon to a value of -9999
    masked_diff, mask_transform = rmask.mask(img, outline.geometry, nodata=-9999)
    mask_meta = img.meta
# create a masked array with numpy setting -9999 as a 'bad' or out-of-range value
masked_diff = np.ma.masked_where(masked_diff == -9999, masked_diff)
with rio.open('{}\\masked_diff.tif'.format(savepath), "w", **mask_meta) as mskdest:
    mskdest.write(masked_diff)

# if no thresholds are set by myconfig they will be determined using Otsu's method
if posthresh == 0 and negthresh == 0:
    posthresh, negthresh = get_threshold('{}\\masked_diff.tif'.format(savepath))
else:
    print('User set thresholds: {}, {}'.format(posthresh, negthresh))

classify_change('{}\\masked_diff.tif'.format(savepath), posthresh, negthresh, '{}\\classified.tif'.format(savepath))
change_polys = getpolygons('{}\\classified.tif'.format(savepath))
pos_change_poly = change_polys[change_polys['raster_val'] == 1]
neg_change_poly = change_polys[change_polys['raster_val'] == 2]
pos_change_area = getarea(pos_change_poly).sum()
neg_change_area = getarea(neg_change_poly).sum()
print('Area of positive change: {:.2f} km²'.format(pos_change_area))
print('Area of negative change: {:.2f} km²'.format(neg_change_area))

# --------------------------------[ PLOTTING ]--------------------------------------

# calculate fig size using aspect ratio of inputs
figaspect = (o_width / o_height)
figmax = 12
if o_width > o_height:
    figwidth, figheight = figmax, (figmax / figaspect)
else:
    figwidth, figheight = (figmax * figaspect) + 1, figmax

# create figure and axes
myCRS = ccrs.UTM(outline.crs.utm_zone)
fig = plt.figure(figsize=(figwidth, figheight))
ax = plt.axes(projection=myCRS)
ax.set_extent([xmin, xmax, ymin, ymax], crs=myCRS)
plt.title(label=maptitle, size=20, pad=20)

# create colormap setting alpha for masked values
mycmap = plt.cm.get_cmap("RdYlBu").copy()
mycmap.set_bad('k', alpha=0)

ax.stock_img()  # displays a (very) low res natural earth background

# display raster
im = ax.imshow(masked_diff[0], cmap=mycmap, vmin=-0.5, vmax=0.5, transform=myCRS, extent=[xmin, xmax, ymin, ymax])

# display polygons
outline_style = {'edgecolor': 'r',
                 'facecolor': 'none',
                 'linewidth': 3.0}
outline_disp = ShapelyFeature(outline['geometry'], myCRS, **outline_style)
ax.add_feature(outline_disp)
outline_patch = mpatches.Patch(label='Area of Interest', **outline_style)

pos_change_style = {'edgecolor': 'none',
                    'facecolor': 'b'}
pos_change_disp = ShapelyFeature(pos_change_poly['geometry'], myCRS, **pos_change_style)
ax.add_feature(pos_change_disp)
pos_change_patch = mpatches.Patch(label='Significant Positive Change', **pos_change_style)

neg_change_style = {'edgecolor': 'none',
                    'facecolor': 'r'}
neg_change_disp = ShapelyFeature(neg_change_poly['geometry'], myCRS, **neg_change_style)
ax.add_feature(neg_change_disp)
neg_change_patch = mpatches.Patch(label='Significant Negative Change', **neg_change_style)

# plot gridlines with labels on top and left
gridlines = ax.gridlines(draw_labels=True)
gridlines.right_labels = False
gridlines.bottom_labels = False

# plot legend

plt.legend(handles=[outline_patch, pos_change_patch, neg_change_patch],
           loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes, borderaxespad=0.1)

resultstext = AnchoredText('Study area size: {:.2f} km²\n'
                           'Area of positive change: {:.2f} km²\n'
                           'Area of negative change: {:.2f} km²'
                           .format(outlinearea, pos_change_area, neg_change_area),
                           loc='upper right', bbox_to_anchor=(1, -0.02), bbox_transform=ax.transAxes,
                           borderpad=0, prop=dict(size=12))
ax.add_artist(resultstext)

# create axes for colorbar plot
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1, axes_class=plt.Axes)

plt.colorbar(im, cax)

plt.subplots_adjust(bottom=0.15)

# show the plot
plt.show()

# save the plot
fig.savefig('{}\\ChangeMap.png'.format(savepath), dpi=300, bbox_inches='tight')
