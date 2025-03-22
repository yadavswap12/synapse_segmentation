"""
This script creates square-shaped tiles from storm images.  

Swapnil 5/23
"""

import numpy as np
import glob
import math
import pickle
import os
import pandas as pd
from tifffile import tifffile

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

# Make directory to store image tiles.
if not os.path.exists(storm_exp_directory + "tiles_size_{}\\" .format(tile_size)):
    os.mkdir(storm_exp_directory + "tiles_size_{}\\" .format(tile_size))

tiles_directory = storm_exp_directory + "tiles_size_{}\\" .format(tile_size) 

# Remove previously present files. 
files = glob.glob(tiles_directory + "*")
for f in files:
    os.remove(f)
    
stromtiffs_directory = storm_exp_directory + "stormtiffs\\"  

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

# Initialize tile count.
tile_count = 0    

# Iterate over individual image numbers.
for img_num in tiles_df["Num_image"].unique():

    # Create a dataframe for particular "Num_img" entry.
    img_df = tiles_df[tiles_df["Num_image"]==img_num]
    
    if channel == '647storm':  

        if ((img_num-1)//10 == 0): stormtiff_file = stromtiffs_directory + "647storm_00" + str(img_num-1) + "_mlist.tiff"
        
        elif (((img_num-1)//10)//10 == 0): stormtiff_file = stromtiffs_directory + "647storm_0" + str(img_num-1) + "_mlist.tiff"
        
        elif ((((img_num-1)//10)//10)//10 == 0): stormtiff_file = stromtiffs_directory + "647storm_" + str(img_num-1) + "_mlist.tiff"
    
    stormtiff_image_array = tifffile.imread(stormtiff_file)    # type(stormtiff_image_array) = numpy.ndarray
    
    # Iterate over all rows in dataframe for given image number.
    for idx in img_df.index:
    
        # Get the tile coordinates and tile ID.
        tile_start_pix_y = math.floor(img_df.loc[idx, 'y(row)'])
        tile_start_pix_x = math.floor(img_df.loc[idx, 'x(column)'])
        tile_id = math.floor(img_df.loc[idx, 'Tile_ID'])
        
        # Slicing the image array according to the tile position.        
        tile = stormtiff_image_array[tile_start_pix_y:tile_start_pix_y+tile_size, tile_start_pix_x:tile_start_pix_x+tile_size]                 
        
        # tile_file = tiles_directory + "tile_" + str(img_num) + str(tile_count+1) + ".data"
        # tile_file = tiles_directory + "tile_" + str(tile_count+1) + ".data"
        tile_file = tiles_directory + "tile_" + str(tile_id) + "image_" + str(img_num) + ".data"                        
        
        # Writting the input and output lists to files for future use (See https://stackabuse.com/reading-and-writing-lists-to-a-file-in-python/ for pickle method)
        with open(tile_file, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(tile, filehandle)
        
        # div = tiles_num//10
        # div = 1            
        div = 1000
        
        if ((tile_count+1)%div==0):
            print("{}th tile is created\n" .format(tile_count+1))
            
        tile_count += 1


































  





