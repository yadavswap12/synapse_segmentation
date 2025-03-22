"""
This script creates a list of square shaped tiles with tile-start coordinates from given list of stormtiff images.
The list is shuffled to create random sequence of tiles.  

Swapnil 5/23
"""

import numpy as np
import glob
import math
import pickle
import os
import pandas as pd    

# Are you collecting training data or test data?
training = True
# training = False    

# Set path to training data files
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
# storm_exp_name = "750storm_no_saturation"

if training: storm_exp_directory = training_data_directory + storm_exp_name + "\\"

else: storm_exp_directory = testing_data_directory + storm_exp_name + "\\" 

# Which channel do you want to analyze? Set experiment name.
# channel = "561storm"
channel = "647storm"    
# channel = "750storm"
base = str(channel)       

# Get the original tile_list file.
if (channel == "561storm"): tile_list_file = storm_exp_directory + "tile_list\\" + "561_ROIs_to_csv.csv"

elif (channel == "647storm"): tile_list_file = storm_exp_directory + "tile_list\\" + "647_ROIs_to_csv.csv"

elif (channel == "750storm"): tile_list_file = storm_exp_directory + "tile_list\\" + "750_ROIs_to_csv.csv" 

# Make a dataframe from csv file containing list of tile coordinates.
tiles_df = pd.read_csv(tile_list_file)

tile_list = []


# Iterate over individual image numbers.
for img_num in tiles_df["Num_image"].unique():

    tile_id = 0
    
    for j in range(image_size[0]//tile_size):
    
        for i in range(image_size[1]//tile_size):
        
            tile_id += 1
        
            tile_start_pix_x = i*tile_size
            tile_start_pix_y = j*tile_size
            
            tile_list.append((tile_start_pix_x, tile_start_pix_y, img_num, tile_id))
            
# Make dataframe out of list of tuples.
df = pd.DataFrame(tile_list, columns = ['x(column)', 'y(row)', 'Num_image', 'Tile_ID'])            
            
# shuffle the DataFrame rows.
# df = df.sample(frac = 1).reset_index(inplace=True, drop=True)
df_shuffled = df.sample(frac = 1).reset_index(drop=True)

# Get the output file for the concatenated dataframe.
df_out_file = storm_exp_directory + "tile_list\\" + "647_new_tiles_full_shuffled.csv" 
# df_out_file = storm_exp_directory + "tile_list\\" + "750_ROIs_to_csv_full_shuffled.csv" 

# Write the concatenated dataframe to .csv file.
df_shuffled.to_csv(df_out_file)        
    
