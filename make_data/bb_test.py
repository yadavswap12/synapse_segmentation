"""
Script to test the correct execution of bounding box formations.  

Swapnil 5/23
"""

import numpy as np
import glob
import math
import pickle
import os
import pandas as pd
from tifffile import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Are you collecting training data or test data?
training = True
# training = False    

# Set path to data files.
expfolder = "C:\\Users\\swapnil.yadav\\Research\\synapse_segmentation\\"

data_directory = expfolder + "make_data\\"

training_data_directory = data_directory + "training_data\\"
testing_data_directory = data_directory + "testing_data\\"


# Get image size.
image_size = (6400, 6400)  

# Get tile size.
tile_size = 400

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

bb_tiles_directory = storm_exp_directory + "bb_tiles_size_{}\\" .format(tile_size)     

bb_tile_list_file = storm_exp_directory + "bounding_boxes\\" + "bb_list_tile_size_{}.csv" .format(tile_size)   
    
# Make a dataframe from csv file containing list of tile coordinates.
bb_tile_df = pd.read_csv(bb_tile_list_file)

tile_list_file = storm_exp_directory + "tile_list\\" + "647_new_tiles_full_shuffled.csv"    
    
# Make a dataframe from csv file containing list of tile coordinates.
df = pd.read_csv(tile_list_file)

# Get total number of tiles.
tiles_num = len(df)

# Set percentage of total data used for training.
training_data_pctg = 90

# Get number of tiles for training out of all tiles.
tiles_train_num = math.ceil(tiles_num*(training_data_pctg/100))    

# Create training and testing dataframes.
# Splitting dataframe by row index.
if training: tiles_df = df.iloc[:tiles_train_num,:]
else: tiles_df = df.iloc[tiles_train_num+1:,:]


# Get tile number for testing tile.
# t_num = 116
t_num = 1

# Get image number for testing tile.
# i_num = 5
i_num = 3  

# Get edge coordinates for test tile.
tile_start_pix_x = tiles_df[(tiles_df['Tile_ID'] == t_num) & (tiles_df['Num_image'] == i_num)]['x(column)'] 
tile_start_pix_y = tiles_df[(tiles_df['Tile_ID'] == t_num) & (tiles_df['Num_image'] == i_num)]['y(row)'] 

# Get the list of old tile_ids overlapping with new tile.
# old_tile_id = bb_tile_df[(bb_tile_df['tile_num'] == t_num) & (bb_tile_df['image_num'] == i_num)]['tile_id'].values[0]
old_tile_id = bb_tile_df[(bb_tile_df['tile_num'] == t_num) & (bb_tile_df['image_num'] == i_num)]['tile_id'].tolist()

print(f"Old tile_id is {old_tile_id}.") 

bb_tile_file = bb_tiles_directory + "bb_tile_" + str(t_num) + "image_" + str(i_num) + ".data"

# Read from .data files. 
with open(bb_tile_file, 'rb') as filehandle:
    bb_tile = pickle.load(filehandle)

tile_file = tiles_directory + "tile_" + str(t_num) + "image_" + str(i_num) + ".data"

# Read from .data files. 
with open(tile_file, 'rb') as filehandle:
    tile = pickle.load(filehandle)

# # Plot tile.
# plt.imshow(tile, cmap='gray')

for bb in bb_tile:
    
    # Get bb in image coordinates.
    (b_x, b_y, b_h, b_w) = bb
    
    # Get bb in tile coordinates.
    b_x = b_x - tile_start_pix_x
    b_y = b_y - tile_start_pix_y 
    
    # Get rectangle coordinates from bb coordinates.
    r_x = b_x - b_w/2.0
    r_y = b_y - b_h/2.0
    r_h = b_h  
    r_w = b_w

    # Plot tile.
    plt.imshow(tile, cmap='gray')    

    # Add the patch to the Axes
    plt.gca().add_patch(Rectangle((r_x,r_y),r_w,r_h,linewidth=1,edgecolor='r',facecolor='none'))
                        
    plt.show()  








































  





