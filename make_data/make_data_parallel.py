"""
This script prepares training and testing data for deep learning segmentation model.  

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


def getOverlap(ss_bb, bb):
    
    """
    Function to get overlap between two bounding boxes.
    Overlap is defined as minimum of the fraction of overlapped area for two bounding boxes. 
    
    ss_bb: Tuple with selective search bb coordinates.
    bb: Tuple with synapse bb coordinates.
    """
    
    # Get ss bounding box coordinates.
    ss_bb_x = ss_bb[0]
    ss_bb_y = ss_bb[1]   
    ss_bb_h = ss_bb[2]   
    ss_bb_w = ss_bb[3]
    ss_bb_x_max = ss_bb_x + ss_bb_w/2.0  
    ss_bb_x_min = ss_bb_x - ss_bb_w/2.0
    ss_bb_y_max = ss_bb_y + ss_bb_h/2.0  
    ss_bb_y_min = ss_bb_y - ss_bb_h/2.0
    ss_bb_area = (ss_bb_x_max-ss_bb_x_min)*(ss_bb_y_max-ss_bb_y_min)     

    # Get synapse bounding box coordinates.
    bb_x = bb[0]
    bb_y = bb[1]   
    bb_h = bb[2]   
    bb_w = bb[3]
    bb_x_max = bb_x + bb_w/2.0  
    bb_x_min = bb_x - bb_w/2.0
    bb_y_max = bb_y + bb_h/2.0  
    bb_y_min = bb_y - bb_h/2.0
    bb_area = (bb_x_max-bb_x_min)*(bb_y_max-bb_y_min)     

    # No overlap condition.
    no_ovrlp_cond = (ss_bb_x_max <= bb_x_min) | (ss_bb_y_max <= bb_y_min) | (bb_x_max <= ss_bb_x_min) | (bb_y_max <= ss_bb_y_min)    
    
    if no_ovrlp_cond:
        
        return 0.0

    else:        
    
        if ss_bb_x_max < bb_x_max:
        
            if ss_bb_x_min < bb_x_min:  
            
                if ss_bb_y_max < bb_y_max:

                    if ss_bb_y_min < bb_y_min:
                    
                        x_min = bb_x_min 
                        x_max = ss_bb_x_max
                        y_min = bb_y_min 
                        y_max = ss_bb_y_max
                            
                    else:    #(ss_bb_y_min >= bb_y_min)
                        x_min = bb_x_min 
                        x_max = ss_bb_x_max
                        y_min = ss_bb_y_min 
                        y_max = ss_bb_y_max

                else:    #(ss_bb_y_max >= bb_y_max)
                
                    if ss_bb_y_min < bb_y_min:
                    
                        x_min = bb_x_min 
                        x_max = ss_bb_x_max
                        y_min = bb_y_min 
                        y_max = bb_y_max
                            
                    else:    #(ss_bb_y_min >= bb_y_min)
                        x_min = bb_x_min 
                        x_max = ss_bb_x_max
                        y_min = ss_bb_y_min 
                        y_max = bb_y_max            
                    
            else:    #(ss_bb_x_min >= bb_x_min)

                if ss_bb_y_max < bb_y_max:

                    if ss_bb_y_min < bb_y_min:
                    
                        x_min = ss_bb_x_min 
                        x_max = ss_bb_x_max
                        y_min = bb_y_min 
                        y_max = ss_bb_y_max
                            
                    else:    #(ss_bb_y_min >= bb_y_min)
                        x_min = ss_bb_x_min 
                        x_max = ss_bb_x_max
                        y_min = ss_bb_y_min 
                        y_max = ss_bb_y_max

                else:    #(ss_bb_y_max >= bb_y_max)
                
                    if ss_bb_y_min < bb_y_min:
                    
                        x_min = ss_bb_x_min 
                        x_max = ss_bb_x_max
                        y_min = bb_y_min 
                        y_max = bb_y_max
                            
                    else:    #(ss_bb_y_min >= bb_y_min)
                        x_min = ss_bb_x_min 
                        x_max = ss_bb_x_max
                        y_min = ss_bb_y_min 
                        y_max = bb_y_max

        else:    #(ss_bb_x_max >= bb_x_max)
        
            if ss_bb_x_min < bb_x_min:  
            
                if ss_bb_y_max < bb_y_max:

                    if ss_bb_y_min < bb_y_min:
                    
                        x_min = bb_x_min 
                        x_max = bb_x_max
                        y_min = bb_y_min 
                        y_max = ss_bb_y_max
                            
                    else:    #(ss_bb_y_min >= bb_y_min)
                        x_min = bb_x_min 
                        x_max = bb_x_max
                        y_min = ss_bb_y_min 
                        y_max = ss_bb_y_max

                else:    #(ss_bb_y_max >= bb_y_max)
                
                    if ss_bb_y_min < bb_y_min:
                    
                        x_min = bb_x_min 
                        x_max = bb_x_max
                        y_min = bb_y_min 
                        y_max = bb_y_max
                            
                    else:    #(ss_bb_y_min >= bb_y_min)
                        x_min = bb_x_min 
                        x_max = bb_x_max
                        y_min = ss_bb_y_min 
                        y_max = bb_y_max            
                    
            else:    #(ss_bb_x_min >= bb_x_min)

                if ss_bb_y_max < bb_y_max:

                    if ss_bb_y_min < bb_y_min:
                    
                        x_min = ss_bb_x_min 
                        x_max = bb_x_max
                        y_min = bb_y_min 
                        y_max = ss_bb_y_max
                            
                    else:    #(ss_bb_y_min >= bb_y_min)
                        x_min = ss_bb_x_min 
                        x_max = bb_x_max
                        y_min = ss_bb_y_min 
                        y_max = ss_bb_y_max

                else:    #(ss_bb_y_max >= bb_y_max)
                
                    if ss_bb_y_min < bb_y_min:
                    
                        x_min = ss_bb_x_min 
                        x_max = bb_x_max
                        y_min = bb_y_min 
                        y_max = bb_y_max
                            
                    else:    #(ss_bb_y_min >= bb_y_min)
                        x_min = ss_bb_x_min 
                        x_max = bb_x_max
                        y_min = ss_bb_y_min 
                        y_max = bb_y_max

        ovrlp_area = (x_max-x_min)*(y_max-y_min) 
        return min(ovrlp_area/ss_bb_area, ovrlp_area/bb_area)

                    
                    
    



def makeInputData(tile, ss_bb_list, resize, img_num, tile_id, input_data_directory):
    
    """
    Function to make input data for deep learning segmentation model.
    
    tile: Tile from which bounding boxes are extracted.
    ss_bb_list: List of bounding boxes from selective search on given tile.
    resize: Tuple with dimensions of resized image.
    """
    
    for ss_bb in ss_bb_list:
    
        # Get bounding box coordinates.
        ss_bb_x = ss_bb[0]
        ss_bb_y = ss_bb[1]   
        ss_bb_h = ss_bb[2]   
        ss_bb_w = ss_bb[3]   
        
        # Get the image portion within the bounding box.
        # tile_bb = tile[ss_bb_y-int(np.ceil(ss_bb_h/2.0)):ss_bb_y+int(np.ceil(ss_bb_h/2.0)), ss_bb_x-int(np.ceil(ss_bb_w/2.0)):ss_bb_x+int(np.ceil(ss_bb_w/2.0))]
        tile_bb = tile[int(ss_bb_y-ss_bb_h/2.0):int(ss_bb_y+ss_bb_h/2.0), int(ss_bb_x-ss_bb_w/2.0):int(ss_bb_x+ss_bb_w/2.0)]
        
        # Resize image portion within the bounding box.  
        tile_bb_rs = cv2.resize(tile_bb, dsize=resize, interpolation=cv2.INTER_CUBIC)

        # Set the filename for input file.
        input_file = input_data_directory + "input_tile_" + str(tile_id) + "_image_" + str(img_num) + "_bb_" + str(int(ss_bb_x)) + "_" + str(int(ss_bb_y)) + "_" + str(int(ss_bb_h)) + "_" + str(int(ss_bb_w)) + ".data"               

        # Writting the input and output lists to files for future use (See https://stackabuse.com/reading-and-writing-lists-to-a-file-in-python/ for pickle method)
        with open(input_file, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(tile_bb_rs, filehandle)
        
        

def makeOutputData(ss_bb_list, bb_list, img_num, tile_id, output_data_directory):
    
    """
    Function to make input data for deep learning segmentation model.
    
    ss_bb_list: List of bounding boxes from selective search on given tile.
    bb_list: List of bounding boxes for synapses in given tile.
    """
    
    for ss_bb in ss_bb_list:
    
        # Initialize output list.
        output = []
        
        # Set bb overlap threshold.
        bb_ovrlp_thrsh = 0.5
        
        # Initialize synapse flag.
        syn_flg = 0
        
        # Initialize maximum overlap for bounding boxes.
        bb_ovrlp_max = 0.0
    
        # Get bounding box coordinates.
        ss_bb_x = ss_bb[0]
        ss_bb_y = ss_bb[1]   
        ss_bb_h = ss_bb[2]   
        ss_bb_w = ss_bb[3]   
        
        for bb in bb_list:
        
            # Get overlap between bounding boxes.
            bb_ovrlp = getOverlap(ss_bb, bb)
            
            if bb_ovrlp > bb_ovrlp_max: bb_ovrlp_max = bb_ovrlp

        if bb_ovrlp_max >= bb_ovrlp_thrsh:
            
            output.append(1)
            output.append(bb_ovrlp_max)
            
        else:

            output.append(0)
            output.append(0.0)

        # Set the filename for input file.
        output_file = output_data_directory + "output_tile_" + str(tile_id) + "_image_" + str(img_num) + "_bb_" + str(int(ss_bb_x)) + "_" + str(int(ss_bb_y)) + "_" + str(int(ss_bb_h)) + "_" + str(int(ss_bb_w)) + ".data"               

        # Writting the input and output lists to files for future use (See https://stackabuse.com/reading-and-writing-lists-to-a-file-in-python/ for pickle method)
        with open(output_file, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(output, filehandle)            
            
            







# Define function to implement selective search.
def makeData(tile, ss_bb_list, bb_list, resize, img_num, tile_id, input_data_directory, output_data_directory, sema):

    """
    Function to make input and output data for deep learning segmentation model.
    
    tile: Tile from which bounding boxes are extracted.
    ss_bb_list: List of bounding boxes from selective search on given tile.
    bb_list: List of bounding boxes for synapses in given tile.
    """
    
    # Process details.
    curr_process = multiprocessing.current_process()
    # parent_process = multiprocessing.parent_process()
    print("Process Name : {} (Daemon : {}), Process Identifier : {}\n".format(curr_process.name, curr_process.daemon, curr_process.pid))

    makeInputData(tile, ss_bb_list, resize, img_num, tile_id, input_data_directory)
    makeOutputData(ss_bb_list, bb_list, img_num, tile_id, output_data_directory)    
        
    # `release` will add 1 to `sema`, allowing other 
    # processes blocked on it to continue
    sema.release()             


if __name__ == "__main__":

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
    
    # Get resize shape for selected search bounding box as input to CNN.
    resize = (72,72)

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

    # Get ss_bb_tiles directory.
    ss_bb_tiles_directory = storm_exp_directory + "ss_bb_tiles_size_{}\\" .format(tile_size)

    # Get bb_tiles directory.    
    bb_tiles_directory = storm_exp_directory + "bb_tiles_size_{}\\" .format(tile_size)    

    # Make directory to store image tiles.
    if not os.path.exists(storm_exp_directory + "input_data_tiles_size_{}_copy2\\" .format(tile_size)):
        os.mkdir(storm_exp_directory + "input_data_tiles_size_{}_copy2\\" .format(tile_size))

    input_data_directory = storm_exp_directory + "input_data_tiles_size_{}_copy2\\" .format(tile_size)
    
    # Remove previously present files. 
    files = glob.glob(input_data_directory + "*")
    for f in files:
        os.remove(f)    
    
    # Make directory to store image tiles.
    if not os.path.exists(storm_exp_directory + "output_data_tiles_size_{}_copy2\\" .format(tile_size)):
        os.mkdir(storm_exp_directory + "output_data_tiles_size_{}_copy2\\" .format(tile_size))

    output_data_directory = storm_exp_directory + "output_data_tiles_size_{}_copy2\\" .format(tile_size)

    # Remove previously present files. 
    files = glob.glob(output_data_directory + "*")
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
    
    # # Use fraction of the dataframe.
    # tiles_df = tiles_df.sample(n=30000, random_state=123)    

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

            # Get the ss_bb_tile filename.                
            ss_bb_tile_file = ss_bb_tiles_directory + "ss_bb_tile_" + str(tile_id) + "image_" + str(img_num) + ".data"       

            # Read from .data files. 
            with open(ss_bb_tile_file, 'rb') as filehandle:
                ss_bb_list = pickle.load(filehandle)
                
            # Get the bb_tile filename.                
            bb_tile_file = bb_tiles_directory + "bb_tile_" + str(tile_id) + "image_" + str(img_num) + ".data"       

            # Read from .data files. 
            with open(bb_tile_file, 'rb') as filehandle:
                bb_list = pickle.load(filehandle)

            # once max_processes are running, the following `acquire` call
            # will block the main process since `sema` has been reduced
            # to 0. This loop will continue only after one or more 
            # previously created processes complete.
            sema.acquire()            

            # Assign process for each tile.
            # process = multiprocessing.Process(target = locsPerTile, args=(h5_file, tile_start_pix_y, tile_start_pix_x, tile_size, max_locs_per_pixel, nm_per_pixel, storm_image_scale, locs_storm_file_name, locs_tile_file_name, locs_storm))
            process = multiprocessing.Process(target = makeData, args=(tile, ss_bb_list, bb_list, resize, img_num, tile_id, input_data_directory, output_data_directory, sema))                
            jobs.append(process)
            process.start()            
                
                
    # In following lines the join method blocks the execution of the main process (the commands in next lines) 
    # until the process whose join method is called terminates. Without the join method, the 
    # main process won't wait until the process gets terminated.        
    for job in jobs:
        
        job.join()                            
            
            
            
            
            
            




                        




