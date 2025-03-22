"""
Test code to plot tiles from given image section.

Swapnil 10/21
"""

import numpy as np
import glob
import math
import pickle
import itertools
import os
import matplotlib.pyplot as plt
import random

# Are you analyzing training data or testing data?
training = True
# training = False

# Set path to training data files
expfolder = "C:\\Users\\swapnil.yadav\\Research\\synapse_segmentation\\"
data_directory = expfolder + "make_data\\"
training_data_directory = data_directory + "training_data\\"
testing_data_directory = data_directory + "testing_data\\"

# storm_exp_name = "561storm"
storm_exp_name = "647storm"
# storm_exp_name = "750storm"                    

if training: storm_exp_directory = training_data_directory + storm_exp_name + "\\" 

else: storm_exp_directory = testing_data_directory + storm_exp_name + "\\"   

# Specify the tile-size of storm image section for training data.
# tile_size = 100
# tile_size = 86
tile_size = 72


old_tiles_directory = storm_exp_directory + "stormtiff_tiles\\"       
 


# Start of the tile ids for tiles used in prediction.
tile_id_start = 342878
    
# Which tile localizations do you want to plot?
# tile_num = 1
# tile_id = 22360
tile_id = 4102

# Get the ith tile.
# tile_file = tiles_directory + "tile_" + str(tile_num) + ".data"
tile_file = old_tiles_directory + "tile_" + str(tile_id) + ".data"
# tile_file = tiles_directory + "tile_" + str(tile_num + tile_id_start - 1) + ".data"
# tile_file = tiles_directory + "tile_" + str(tile_id - tile_id_start + 1) + ".data"

# Read from .data files. 
with open(tile_file, 'rb') as filehandle:
    tile = pickle.load(filehandle)
    

         
    
# Plot ith tile.
plt.imshow(tile, cmap='gray')
plt.colorbar()
plt.gca().xaxis.tick_top()
# plt.legend(labels=['tile'])
# plt.title("tile_{}" .format(tile_num), pad=30.0)
plt.title("tile_{}" .format(tile_id), pad=30.0)
# plt.title("tile_{}" .format(i), y=1.08)
# plt.show()

