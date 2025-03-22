"""
This script creates a dataframe having bounding box coordinates, tile number etc.  

Swapnil 5/23
"""

import numpy as np
import glob
import math
import pickle
import os
import pandas as pd

from get_bb import getBB
from get_bb_final import getBBFinal

# Are you collecting training data or test data?
training = True
# training = False    

# Set path to data files.
expfolder = "C:\\Users\\swapnil.yadav\\Research\\synapse_segmentation\\"

data_directory = expfolder + "make_data\\"

training_data_directory = data_directory + "training_data\\"
testing_data_directory = data_directory + "testing_data\\"

# storm_exp_name = "561storm"
storm_exp_name = "647storm"    
# storm_exp_name = "750storm"

if training: exp_directory = training_data_directory + storm_exp_name + "\\"

else: exp_directory = testing_data_directory + storm_exp_name + "\\"

clust_pix_list_directory = exp_directory + "cluster_pixel_lists\\"

# Make directory to store bounding box data.
if not os.path.exists(exp_directory + "bounding_boxes\\"):
    os.mkdir(exp_directory + "bounding_boxes\\")

bb_directory = exp_directory + "bounding_boxes\\"    

# Get image size.
image_size = (6400, 6400)  

# Get tile size.
tile_size = 400

# Set bounding box buffer size (in pixels).
# bb_buffer = 10 
bb_buffer = 1 

# Initialize dataframe list.
df_list = []

# Iterate over all the files. 
files = glob.glob(clust_pix_list_directory + "*.txt")

i = 0

for file in files:

    # Get bounding box for given pixel-list.
    b_x, b_y, b_h, b_w, tile_id, image_num = getBB(file, bb_buffer)
    
    # For given bounding box, get a list of tile numbers and divided bounding boxes within those tiles.
    bb_list = getBBFinal(b_x, b_y, b_h, b_w, image_size, tile_size)
    
    # Make dataframe out of list of tuples.
    df = pd.DataFrame(bb_list, columns =['b_x', 'b_y', 'b_h', 'b_w', 'tile_num'])
    df['tile_id'] = tile_id
    df['image_num'] = image_num
    
    
    # Add the dataframe to list.
    df_list.append(df)

    if (i%100 == 0):
        print("{}th pixel-list is processed" .format(i))
    
    i += 1
    
# Create a concatenated dataframe form dataframe list. 
df = pd.concat(df_list, ignore_index=True)

# Get the output file for the concatenated dataframe.
df_out_file = bb_directory + "bb_list_tile_size_{}.csv" .format(tile_size)

# Write the concatenated dataframe to .csv file.
df.to_csv(df_out_file)                






# def make_bb(clust_pix_list_directory, bb_buffer, image_size, tile_size):
    # """
    # Function to create dataframe having bounding box coordinates and tile number.
    
    # clust_pix_list_directory: path to directory with all pixel-lists.
    # bb_buffer: buffer space (in number of pixels) between bounding box and nearest object boundary.
    # image_size: tuple giving image size in pixels.
    # tile_size: size of square-shaped tile in pixels.
    # """
    
    # # Initialize dataframe list.
    # df_list = []
    
    # # Iterate over all the files. 
    # files = glob.glob(clust_pix_list_directory + "*.txt")
    
    # for file in files:
    
        # # Get bounding box for given pixel-list.
        # b_x, b_y, b_h, b_w = get_bb(file, bb_buffer)
        
        # # For given bounding box, get a list of tile numbers and divided bounding boxes within those tiles.
        # bb_list = get_bb_final(b_x, b_y, b_h, b_w, image_size, tile_size)
        
        # # Make dataframe out of list of tuples.
        # df = pd.DataFrame(bb_list, columns =['b_x', 'b_y', 'b_h', 'b_w', 'tile_num'])
        
        # # Add the dataframe to list.
        # df_list.append(clstr_pix_list_df)  
        
    # # Create a concatenated dataframe form dataframe list. 
    # df = pd.concat(df_list, ignore_index=True)

    # # Get the output file for the concatenated dataframe.
    # df_out_file = bb_directory + "bb_list_tile_size_{}.csv" .format(tile_size)

    # # Write the concatenated dataframe to .csv file.
    # df.to_csv(df_out_file)                
        
        
        
    
        
    
    






