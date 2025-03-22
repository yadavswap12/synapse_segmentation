"""
This script creates bounding box coordinates given the pixel-list of the object.  

Swapnil 5/23
"""

import numpy as np
import glob
import math
import pickle
import os
import pandas as pd



def getBB(clstr_pix_list_file, bb_buffer):

    """
    function to get bounding box coordinates around the object from the pixel-list of the object.

    clstr_pix_list_file: filename for pixel-list.
    bb_buffer: buffer space (in number of pixels) between bounding box and nearest object boundary.
    """

    # Make a dataframe from .txt file containing list of pixel coordinates.
    clstr_pix_list_df = pd.read_csv(clstr_pix_list_file)
    
    # Subtract 1 from all pixel coordinates.
    # This is because pixel coordinates in pixel list provided start from 1 while those in our analysis always start from 0.
    clstr_pix_list_df[["x(row)","y(column)"]] = clstr_pix_list_df[["x(row)","y(column)"]].subtract(1)

    # Get minimum and maximum x,y coordinates from pixel-list.
    x_min = clstr_pix_list_df['y(column)'].min()
    x_max = clstr_pix_list_df['y(column)'].max() 
    y_min = clstr_pix_list_df['x(row)'].min() 
    y_max = clstr_pix_list_df['x(row)'].max()

    # Get bounding box with buffer space.
    x_min_buffer = x_min - bb_buffer
    x_max_buffer = x_max + bb_buffer   
    y_min_buffer = y_min - bb_buffer   
    y_max_buffer = y_max + bb_buffer

    b_x = (x_min_buffer + x_max_buffer)/2.0
    b_y = (y_min_buffer + y_max_buffer)/2.0 
    b_h = (y_max_buffer - y_min_buffer)
    b_w = (x_max_buffer - x_min_buffer)
    
    # Get tile_id.
    tile_id = clstr_pix_list_df['TileID'].min()
    image_num = clstr_pix_list_df['Num_image'].min()    
    
    return (b_x, b_y, b_h, b_w, tile_id, image_num)
    
    
    
    


    




    
