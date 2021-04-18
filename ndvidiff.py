import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import rasterio as rio
import rasterio.mask
from cartopy.feature import ShapelyFeature

# --------------------------------[ FUNCTIONS ]--------------------------------------

def calc_ndvi(nir,red):
    '''Calculate NDVI from integer arrays'''
    nir = nir.astype('float32')
    red = red.astype('float32')
    ndvi = np.where(
        (nir+red) == 0.,
        0,
        (nir - red) / (nir + red)
    )  # note this operation will trigger divide by zero errors, but the function will complete
    return ndvi

# --------------------------------[ DATASETS ]--------------------------------------

# load Landsat data
with rio.open('data_files\\LT05_L1TP_182024_19910513_20180225_01_T1_B3.TIF') as img:
    img1red = img.read()  # image 1 visible red
    print(img.crs)
    rastcrs = img.crs  # take crs from the image for use with polygons
    xmin, ymin, xmax, ymax = img.bounds # taking image bounds from landsat image,
    # need to figure out how to make a window/subset using polygon bounds

with rio.open('data_files\\LT05_L1TP_182024_19910513_20180225_01_T1_B4.TIF') as img:
    img1nir = img.read()  # image 1 nir

with rio.open('data_files\\LT05_L1TP_182024_19860531_20170217_01_T1_B3.TIF') as img:
    img2red = img.read()  # image 2 visible red

with rio.open('data_files\\LT05_L1TP_182024_19860531_20170217_01_T1_B4.TIF') as img:
    img2nir = img.read()  # image 2 nir

# load the polygon and set crs to same as the Landsat data
outline = gpd.read_file('data_files\\ExclusionZone.shp').to_crs(rastcrs)

# --------------------------------[ BAND MATHS ]--------------------------------------

ndvi1 = calc_ndvi(img1nir,img1red)
ndvi2 = calc_ndvi(img2nir,img2red)
# ndvidiff = (ndvi1-ndvi2)

# --------------------------------[ PLOTTING ]--------------------------------------

# create figure and axes
myCRS = ccrs.UTM(35)
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=myCRS)
ax.set_extent([xmin, xmax, ymin, ymax], crs=myCRS)

# display raster
ax.imshow(ndvi2[0], cmap='viridis', vmin=0, vmax=1, transform=myCRS, extent=[xmin, xmax, ymin, ymax])

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