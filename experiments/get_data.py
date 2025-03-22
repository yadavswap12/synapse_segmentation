"""
Function to get the data from all the files in given path.
Returns a list with each element of list being data from single file in given file path.

data_path - full path of directory where data is located.

data_type = A string argument specifying the data type.
data_type = "tiles" for image tile data.
data_type = "locs" for localization data.   

Swapnil 10/21
"""

import numpy as np
import glob
import math
import pickle
import os
import re
import random

def getData(data_path, data_type, data_size):

    if (data_type == "input"):
    
        data = []

        # Initialize tile-id list.
        tile_id_list = []
        
        # Get number of files present in given path.
        total_files = len(glob.glob(data_path + "*.data"))
        
        # Get files in sorted order according to number in filename.
        # See: https://stackoverflow.com/questions/6773584/how-is-pythons-glob-glob-ordered  
        tile_files = sorted(glob.glob1(data_path, "*.data"), key=lambda x:float(re.findall("(\d+)",x)[0]))        
        
        # for i in range(1, total_files+1):
        
            # tile_file = data_path + "tile_" + str(i) + ".data"
            
            # # Read from the file and save it to an array 
            # with open(tile_file, 'rb') as filehandle:
                # tile = pickle.load(filehandle)
            
            # data.append(tile)
            
        # for tile_file in tile_files[:data_size]:
        random.seed(123)
        for tile_file in random.sample(tile_files, data_size):
        
                    
            # Read from the file and save it to an array 
            with open(data_path + tile_file, 'rb') as filehandle:
                tile = pickle.load(filehandle)
            
            # Get the number present in the filename.
            tile_id = re.search(r'\d+', tile_file).group(0)                
            
            data.append(tile)
            tile_id_list.append(tile_id)
            
            
    elif (data_type == "output"):

        data = []
        
        # Initialize tile-id list.
        tile_id_list = []        
        
        # Get number of files present in given path.
        total_files = len(glob.glob(data_path + "*.data"))
        
        # Get files in sorted order according to number in filename.
        # See: https://stackoverflow.com/questions/6773584/how-is-pythons-glob-glob-ordered  
        num_locs_tile_files = sorted(glob.glob1(data_path, "*.data"), key=lambda x:float(re.findall("(\d+)",x)[0]))     
        
        # for i in range(1, total_files+1):
        
            # num_locs_tile_file_name = data_path + "storm_num_locs_tile_" + str(i) + ".data"                
            
            # # Read from the file and save it to an array 
            # with open(num_locs_tile_file_name, 'rb') as filehandle:
                # num_locs_tile = pickle.load(filehandle)
            
            # data.append(num_locs_tile)
            
        # for num_locs_tile_file in num_locs_tile_files[:data_size]:
        random.seed(123)
        for num_locs_tile_file in random.sample(num_locs_tile_files, data_size):
        
                    
            # Read from the file and save it to an array 
            with open(data_path + num_locs_tile_file, 'rb') as filehandle:
                num_locs_tile = pickle.load(filehandle)
                
            # Get the number present in the filename.
            tile_id = re.search(r'\d+', num_locs_tile_file).group(0)                    
            
            data.append(num_locs_tile)
            tile_id_list.append(tile_id)            
                                              
            
    return data, tile_id_list         
    
        