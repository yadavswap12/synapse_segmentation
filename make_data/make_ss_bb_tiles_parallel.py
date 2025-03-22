"""
This script collects selective search bounding boxes for given tile.  

Swapnil 5/23
"""

import numpy as np
import glob
import math
import pickle
import os
import pandas as pd
from tifffile import tifffile
import selectivesearch
import matplotlib.patches as mpatches
import cv2
import multiprocessing
from multiprocessing import Semaphore

# Define function to implement selective search.
def selectiveSearchTiles(ss_model, tile, old_tile_size, ss_bb_tiles_directory, img_num, tile_id, sema):

    """
    Function to get bounding boxes of segmented regions in tile using selective search algorithm.
    
    ss_model: selective search algorithm type.
    tile: tile on which selective search is run.
    old_tile_size: tile size from  localization prediction project.
    ss_bb_tiles_directory: directory to store bounding boxes from selective search.
    img_num: Image number for given tile.
    tile_id : Tile number for given tile.
    """
    
    # Process details.
    curr_process = multiprocessing.current_process()
    # parent_process = multiprocessing.parent_process()
    print("Process Name : {} (Daemon : {}), Process Identifier : {}\n".format(curr_process.name, curr_process.daemon, curr_process.pid))        

    ss_bb_list = []            
        
    if ss_model == 'ss_python':
    
        # # If necessary reshape input data from tiles to a shape the opencv expects.
        # tile = np.reshape(tile, (tile.shape[0], tile.shape[1], 3))
        num_repeats = 3
        tile = np.dstack([tile]*num_repeats)

        # # loading image.
        # img = tile

        # # perform selective search
        # img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.8, min_size=1)
        
        # perform selective search
        img_lbl, regions = selectivesearch.selective_search(tile, scale=500, sigma=0.8, min_size=1)                
            
        candidates = set()
        for r in regions:
        
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
                
            # excluding regions smaller than pixel threshold size for bounding box.
            if r['size'] < 50:                
            # if r['size'] < 1:    
                continue
                
            # exclude distorted rects.
            x, y, w, h = r['rect']
            if w == 0 or h==0:    
                continue
                
            # Exclude rectangles larger than typical synapse.
            if w > old_tile_size  or h > old_tile_size:    
                continue                        
            
            candidates.add(r['rect'])
            
            # Get selective_search bounding box coordinates.
            (ss_b_x, ss_b_y, ss_b_h, ss_b_w) = (x+w/2.0, y+h/2.0, h, w)                
            ss_bb_list.append((ss_b_x, ss_b_y, ss_b_h, ss_b_w))

    ss_bb_tile_file = ss_bb_tiles_directory + "ss_bb_tile_" + str(tile_id) + "image_" + str(img_num) + ".data"                        
    
    # Writting the input and output lists to files for future use (See https://stackabuse.com/reading-and-writing-lists-to-a-file-in-python/ for pickle method)
    with open(ss_bb_tile_file, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(ss_bb_list, filehandle)
        
    # `release` will add 1 to `sema`, allowing other 
    # processes blocked on it to continue
    sema.release()             


if __name__ == "__main__":

    # Choose selective search model.
    # ss_model = 'ss_opencv'
    ss_model = 'ss_python'

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

    # Get old tile size from localization prediction project.
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

    # Get tiles directory.
    tiles_directory = storm_exp_directory + "tiles_size_{}\\" .format(tile_size)         

    # Make directory to store image tiles.
    if not os.path.exists(storm_exp_directory + "ss_bb_tiles_size_{}\\" .format(tile_size)):
        os.mkdir(storm_exp_directory + "ss_bb_tiles_size_{}\\" .format(tile_size))

    ss_bb_tiles_directory = storm_exp_directory + "ss_bb_tiles_size_{}\\" .format(tile_size) 

    # Remove previously present files. 
    files = glob.glob(ss_bb_tiles_directory + "*")
    for f in files:
        os.remove(f)
        

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

    # Initialize tile count.
    tile_count = 0

    # Set Maximum number of parallel processes.
    # max_processes = 50

    # See: https://stackoverflow.com/questions/20039659/python-multiprocessings-pool-process-limit/20039847#:~:text=Theoretically%20there%20is%20no%20limit,the%20running%20out%20of%20memory.    
    # max_processes = 50
    max_processes = multiprocessing.cpu_count() - 1  

    # Setup process queue.
    jobs = []
    # process_count = 0
    # results = multiprocessing.Queue()

    # See: https://stackoverflow.com/questions/20886565/using-multiprocessing-process-with-a-maximum-number-of-simultaneous-processes
    sema = Semaphore(max_processes)         

    # Iterate over individual image numbers.
    for img_num in tiles_df["Num_image"].unique():

        # Create a dataframe for particular "Num_img" entry.
        img_df = tiles_df[tiles_df["Num_image"]==img_num]
        bb_tile_img_df = bb_tile_df[bb_tile_df["image_num"]==img_num]
        
        # Iterate over all rows in dataframe for given image number.
        for idx in img_df.index:
        
            # Get the tile coordinates and tile ID.
            tile_id = math.floor(img_df.loc[idx, 'Tile_ID'])
            
            # Get the tile filename.
            tile_file = tiles_directory + "tile_" + str(tile_id) + "image_" + str(img_num) + ".data"

            # Read from .data files. 
            with open(tile_file, 'rb') as filehandle:
                tile = pickle.load(filehandle)                
                
            # once max_processes are running, the following `acquire` call
            # will block the main process since `sema` has been reduced
            # to 0. This loop will continue only after one or more 
            # previously created processes complete.
            sema.acquire()            

            # Assign process for each tile.
            # process = multiprocessing.Process(target = locsPerTile, args=(h5_file, tile_start_pix_y, tile_start_pix_x, tile_size, max_locs_per_pixel, nm_per_pixel, storm_image_scale, locs_storm_file_name, locs_tile_file_name, locs_storm))
            process = multiprocessing.Process(target = selectiveSearchTiles, args=(ss_model, tile, old_tile_size, ss_bb_tiles_directory, img_num, tile_id, sema))                
            jobs.append(process)
            process.start()            
                
                
    # In following lines the join method blocks the execution of the main process (the commands in next lines) 
    # until the process whose join method is called terminates. Without the join method, the 
    # main process won't wait until the process gets terminated.        
    for job in jobs:
        
        job.join()                            
            
            
            
            
            
            




                        




