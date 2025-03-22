"""
Script to create parallel processes for making dictionary of bounding box coordinates in specified image of stormtiff image list.
The dictionary file name indicates the stormtiff image number, thus every stormtiff image has separate dictionary. 
Every key of the dictionary specifies coordinates of a particular tile edge in given stormtiff image and the 
corresponding value gives list of localizations within this tile in the form [np.array([x0,y0]), np.array([x1,y1]), ....].

This script uses multiprocessing module. 
        
Swapnil 2/22
"""

import numpy as np
import storm_analysis.sa_library.sa_h5py as saH5Py
import pickle
import itertools
import math
import os
from tifffile import tifffile
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib
# from matplotlib.offsetbox import AnchoredText
import multiprocessing

from locs_per_storm_image import locsPerStormImage
from multiprocessing import Semaphore

if __name__ == "__main__":

    # Set path to data files.
    expfolder = "C:\\Users\\swapnil.yadav\\Research\\loc_prediction\\storm\\project_16\\"
    XML_folder = expfolder + "XMLs\\"
    data_directory = expfolder + "make_data\\"
    training_data_directory = data_directory + "training_data\\"
    # testing_data_directory = data_directory + "testing_data\\"
    # storm_exp_name = "561storm"
    storm_exp_name = "647storm"
    # storm_exp_name = "750storm"    
    storm_exp_directory = training_data_directory + storm_exp_name + "\\"
    # storm_exp_directory = testing_data_directory + storm_exp_name + "\\"    
    molecule_lists_directory = storm_exp_directory + "molecule_lists\\"
    
    # tile_list_file = storm_exp_directory + "tile_list\\" + "561_ROIs_to_csv_8k.csv"
    # tile_list_file = storm_exp_directory + "tile_list\\" + "647_ROIs_to_csv_8k.csv"
    # tile_list_file = storm_exp_directory + "tile_list\\" + "647_ROIs_to_csv_14k.csv"
    tile_list_file = storm_exp_directory + "tile_list\\" + "647_ROIs_to_csv.csv"                
    # tile_list_file = storm_exp_directory + "tile_list\\" + "647_ROIs_to_csv_from_3k_14k_shuffled.csv"            
    # tile_list_file = storm_exp_directory + "tile_list\\" + "750_ROIs_to_csv_mixed_first_70.csv" 
    # tile_list_file = storm_exp_directory + "tile_list\\" + "750_ROIs_to_csv_mixed_first_100.csv"
    # tile_list_file = storm_exp_directory + "tile_list\\" + "750_ROIs_to_csv_14k.csv"
    # tile_list_file = storm_exp_directory + "tile_list\\" + "750_ROIs_to_csv.csv"     
    # tile_list_file = storm_exp_directory + "tile_list\\" + "750_ROIs_to_csv_14k_shuffled.csv"        

    # Set tile size for square shaped tile.
    # tile_size = 86
    # tile_size = 84    
    tile_size = 72        

    # Specify ROI extension string for data directory.
    # roi = "_ROIs_mixed_first_70"        
    # roi = "_ROIs_mixed_first_100"
    # roi = "_ROIs_14k" 
    # roi = "_ROIs_from_3k_14k_shuffled"
    roi = "_ROIs"                
    
    # If does not exists, create a directory for localization dictionary.
    if not os.path.exists(storm_exp_directory + "locs_dictionary_tilesize_" + str(tile_size) + roi + "\\"):
        os.mkdir(storm_exp_directory + "locs_dictionary_tilesize_" + str(tile_size) + roi + "\\")
        
    locs_dict_directory = storm_exp_directory + "locs_dictionary_tilesize_" + str(tile_size) + roi + "\\"

    # Make a dataframe from csv file containing list of tile coordinates.
    df = pd.read_csv(tile_list_file)

    print("total tiles are {}" .format(len(df)))

    # Initialize maximum localizations per tile.
    max_locs_per_tile = 0

    # Set scaling factor between raw image and stormtiff image.
    storm_image_scale = int(10)
    
    # Set Maximum number of parallel processes.
    # See: https://stackoverflow.com/questions/20039659/python-multiprocessings-pool-process-limit/20039847#:~:text=Theoretically%20there%20is%20no%20limit,the%20running%20out%20of%20memory.    
    # max_processes = 50
    max_processes = multiprocessing.cpu_count() - 1    
    
    # Setup process queue.
    jobs = []
    # process_count = 0
    # results = multiprocessing.Queue()

    # See: https://stackoverflow.com/questions/20886565/using-multiprocessing-process-with-a-maximum-number-of-simultaneous-processes
    sema = Semaphore(max_processes)        
    
    # Iterate over individual image numbers in dataframe.
    for img_num in df["Num_image"].unique():

        # Create a dataframe for particular "Num_img" entry.
        img_df = df[df["Num_image"]==img_num]

        # Get molecule list file for given image number.        
        if ((img_num-1)//10 == 0): h5_file = molecule_lists_directory + storm_exp_name + "_00" + str(img_num-1) + "_mlist" + ".hdf5"
        
        elif (((img_num-1)//10)//10 == 0): h5_file = molecule_lists_directory + storm_exp_name + "_0" + str(img_num-1) + "_mlist" + ".hdf5"
        
        elif ((((img_num-1)//10)//10)//10 == 0): h5_file = molecule_lists_directory + storm_exp_name + "_" + str(img_num-1) + "_mlist" + ".hdf5"
        
        # Create a dictionary filename.
        locs_dict_file_name = locs_dict_directory + "locs_dict_img_" + str(img_num) + ".data"
        
        # once max_processes are running, the following `acquire` call
        # will block the main process since `sema` has been reduced
        # to 0. This loop will continue only after one or more 
        # previously created processes complete.
        sema.acquire()         
        
        # Assign process for each tile
        process = multiprocessing.Process(target = locsPerStormImage, args=(h5_file, tile_size, storm_image_scale, img_df, locs_dict_file_name, sema))
        jobs.append(process)
        process.start()
        
    # In following lines the join method blocks the execution of the main process (the commands in next lines) 
    # until the process whose join method is called terminates. Without the join method, the 
    # main process won't wait until the process gets terminated.        
    for job in jobs:
        
        job.join()        