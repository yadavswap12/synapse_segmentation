"""
This script creates a full pixel list of pixels from all the clusters (connected components) from 
given set of tiles.  

Swapnil 2/22
"""

import numpy as np
import glob
import math
import pickle
import os
import pandas as pd

# Set path to data files
expfolder = "C:\\Users\\swapnil.yadav\\Research\\loc_prediction\\storm\\project_15\\signal_and_loc_density_analysis\\"
# XML_folder = expfolder + "XMLs\\"
data_directory = expfolder + "make_data\\"
# storm_exp_name = "561storm"
# storm_exp_name = "647storm"
storm_exp_name = "750storm"
storm_exp_directory = data_directory + storm_exp_name + "\\"     

# Get the cluster pixel_list directory.
project_folder = "C:\\Users\\swapnil.yadav\\Research\\loc_prediction\\storm\\project_15\\"

# clust_pix_list_directory = storm_exp_directory + "cluster_pixel_lists_tile1_to_tile10000\\"
clust_pix_list_directory = project_folder + "make_data\\training_data\\750storm\\cluster_pixel_lists\\"

# tile_list_file = storm_exp_directory + "tile_list\\" + "561_ROIs_to_csv.csv"
# tile_list_file = storm_exp_directory + "tile_list\\" + "647_ROIs_to_csv.csv"
tile_list_file = storm_exp_directory + "750_ROIs_to_csv_img1_trunc_2.csv"

# Make a dataframe from csv file containing list of tile coordinates.
df = pd.read_csv(tile_list_file)

tile_list = df["Tile_ID"].tolist()

# Initialize dataframe list.
df_list = []

# # Get all the cluster pixel-list files. 
# files = glob.glob(clust_pix_list_directory + "*.txt")
# for file in files:
    # # Make a dataframe from .txt file containing list of pixel coordinates.
    # clstr_pix_list_df = pd.read_csv(file)
    
    # # Add the dataframe to list.
    # df_list.append(clstr_pix_list_df)
    
tile_count = 0    

for tile in tile_list:

    # # Get the cluster-pixel-list file.
    if ((tile)//10 == 0): clstr_pix_list_file = clust_pix_list_directory + "Pix_" + "00000" + str(tile) + ".txt"
    
    elif (((tile)//10)//10 == 0): clstr_pix_list_file = clust_pix_list_directory + "Pix_" + "0000" + str(tile) + ".txt"
    
    elif ((((tile)//10)//10)//10 == 0): clstr_pix_list_file = clust_pix_list_directory + "Pix_" + "000" + str(tile) + ".txt"

    elif (((((tile)//10)//10)//10)//10 == 0): clstr_pix_list_file = sclust_pix_list_directory + "Pix_" + "00" + str(tile) + ".txt"            

    elif ((((((tile)//10)//10)//10)//10)//10 == 0): clstr_pix_list_file = clust_pix_list_directory + "Pix_" + "0" + str(tile) + ".txt"            

    elif (((((((tile)//10)//10)//10)//10)//10)//10 == 0): clstr_pix_list_file = clust_pix_list_directory + "Pix_" + str(tile) + ".txt"            

    # Make a dataframe from .txt file containing list of pixel coordinates.
    clstr_pix_list_df = pd.read_csv(clstr_pix_list_file)
    
    # Add the dataframe to list.
    df_list.append(clstr_pix_list_df)
    
    tile_count += 1
    
    if (tile_count%100 == 0):
        print("{}th Pix file is analyzed." .format(tile_count))


# # Get all the cluster pixel-list files. 
# files = glob.glob(clust_pix_list_directory + "*.txt")
# for file in files:
    # # Make a dataframe from .txt file containing list of pixel coordinates.
    # clstr_pix_list_df = pd.read_csv(file)
    
    # # Add the dataframe to list.
    # df_list.append(clstr_pix_list_df)     

# Create a concatenated dataframe form dataframe list. 
df_combined = pd.concat(df_list, ignore_index=True)

# Get the output file for the concatenated dataframe.
df_out_file = storm_exp_directory + "pix_list_img1_trunc_2.csv"

# Write the concatenated dataframe to .csv file.
df_combined.to_csv(df_out_file)        
    
