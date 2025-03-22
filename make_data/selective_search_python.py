"""
Script to test 'selective search' implementation in opencv python.

See: https://github.com/AlpacaTechJP/selectivesearch  

Swapnil 5/23
"""

import numpy as np
import pickle
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# import cv2

# Are you collecting training data or test data?
training = True
# training = False    

# Set path to data files.
expfolder = "C:\\Users\\Swapnil\\Research\\synapse_segmentation\\"

data_directory = expfolder + "make_data\\"

training_data_directory = data_directory + "training_data\\"
testing_data_directory = data_directory + "testing_data\\"

# Get image size.
image_size = (6400, 6400)  

# Get tile size.
tile_size = 400

# Get old tile size.
old_tile_size = 72

# storm_exp_name = "561storm"
storm_exp_name = "647storm"    
# storm_exp_name = "750storm"

if training: storm_exp_directory = training_data_directory + storm_exp_name + "\\"

else: storm_exp_directory = testing_data_directory + storm_exp_name + "\\"

# Which channel do you want to analyze? Set experiment name.
# channel = "561storm"
channel = "647storm"    
# channel = "750storm"
base = str(channel)  

tiles_directory = storm_exp_directory + "tiles_size_{}\\" .format(tile_size)

# Get tile number for testing tile.
# t_num = 116
t_num = 10

# Get image number for testing tile.
# i_num = 5
# i_num = 3 
i_num = 31

tile_file = tiles_directory + "tile_" + str(t_num) + "image_" + str(i_num) + ".data"

# Read from .data files. 
with open(tile_file, 'rb') as filehandle:
    tile = pickle.load(filehandle)
    
# # If necessary reshape input data from tiles to a shape the opencv expects.
# tile = np.reshape(tile, (tile.shape[0], tile.shape[1], 3))

num_repeats = 3
tile = np.dstack([tile]*num_repeats)

print(f"shape of tile is {tile.shape}.")


# loading astronaut image
# img = skimage.data.astronaut()
# img = cv2.imread('download_grey.jpg')
img = tile


# perform selective search
img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.8, min_size=1)
    
print(f'total regions are {len(regions)}')    

candidates = set()
for r in regions:

    # excluding same rectangle (with different segments)
    if r['rect'] in candidates:
        continue
        
    # excluding regions smaller than 2000 pixels
    # if r['size'] < 2000:
    if r['size'] < 50:    
    # if r['size'] < 1:    
        continue
        
    # distorted rects
    x, y, w, h = r['rect']
    if w == 0 or h==0:    
        continue

    # Exclude rectangles larger than typical synapse.
    if w > old_tile_size  or h > old_tile_size:    
        continue    

    # if w / h > 1.2 or h / w > 1.2:
        # continue
    candidates.add(r['rect'])
    
print(f'total regions after filtering are {len(candidates)}')        

# draw rectangles on the original image
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(img)
for x, y, w, h in candidates:
    # print(x, y, w, h)
    rect = mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)

plt.show()

 