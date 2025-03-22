"""
The script to divide original bounding box into bounding boxes within individual tiles.  

Swapnil 5/23
"""

import numpy as np
import glob
import math
import pickle
import os
import pandas as pd

def getTileNum(x, y, image_size, tile_size):
    
    """
    function to get tile number within which given point lies.
    
    (x,y): coordinates of the point.
    image_size: tuple giving image size in pixels.
    tile_size: size of square-shaped tile in pixels.
    """
    
    return (y//tile_size)*(image_size[0]/tile_size) + (x//tile_size) + 1


def getBBFinal(b_x, b_y, b_h, b_w, image_size, tile_size):

    """
    function to divide bounding box to multiple boxes within individual tiles.

    b_x, b_y, b_h, b_w: bounding box coordinates.
    image_size: tuple giving image size in pixels.
    tile_size: size of square-shaped tile in pixels.
    """
    
    # Get coordinates of bounding box corners.
    lt_x = b_x-b_w/2.0
    lt_y = b_y-b_h/2.0 
    rt_x = b_x+b_w/2.0 
    rt_y = b_y-b_h/2.0 
    lb_x = b_x-b_w/2.0 
    lb_y = b_y+b_h/2.0 
    rb_x = b_x+b_w/2.0 
    rb_y = b_y+b_h/2.0

    # Get tile number for each bounding box corner.
    lt_tile_num = getTileNum(lt_x, lt_y, image_size, tile_size)
    rt_tile_num = getTileNum(rt_x, rt_y, image_size, tile_size)
    lb_tile_num = getTileNum(lb_x, lb_y, image_size, tile_size)
    rb_tile_num = getTileNum(rb_x, rb_y, image_size, tile_size)
    
    # Get tile edge coordinates.
    lt_tile_b_y = ((lt_tile_num-1)//(image_size[0]/tile_size) + 1)*tile_size
    lt_tile_r_x = ((lt_tile_num-1)%(image_size[0]/tile_size) + 1)*tile_size
    rt_tile_l_x = ((rt_tile_num-1)%(image_size[0]/tile_size))*tile_size
    rt_tile_b_y = ((rt_tile_num-1)//(image_size[0]/tile_size) + 1)*tile_size    
    lb_tile_t_y = ((lb_tile_num-1)//(image_size[0]/tile_size))*tile_size
    lb_tile_r_x = ((lb_tile_num-1)%(image_size[0]/tile_size) + 1)*tile_size
    rb_tile_t_y = ((rb_tile_num-1)//(image_size[0]/tile_size))*tile_size
    rb_tile_l_x = ((rb_tile_num-1)%(image_size[0]/tile_size))*tile_size
    
    if lt_tile_num==rt_tile_num==lb_tile_num==rb_tile_num:
        return [(b_x, b_y, b_h, b_w, lt_tile_num)]
        
    elif lt_tile_num==rt_tile_num:
        # For top tile.
        top_tile_lt_x = lt_x 
        top_tile_lt_y = lt_y         
        
        top_tile_rt_x = rt_x
        top_tile_rt_y = rt_y 
        
        top_tile_lb_x = top_tile_lt_x 
        top_tile_lb_y = lt_tile_b_y
        
        top_tile_rb_x = top_tile_rt_x 
        top_tile_rb_y = top_tile_lb_y
        
        top_tile_b_x = (top_tile_lt_x + top_tile_rt_x)/2.0
        top_tile_b_y = (top_tile_lt_y + top_tile_lb_y)/2.0
        top_tile_b_h = top_tile_lb_y - top_tile_lt_y
        top_tile_b_w = top_tile_rt_x - top_tile_lt_x
        

        # For bottom tile.
        bottom_tile_lt_x = top_tile_lt_x 
        bottom_tile_lt_y = top_tile_lb_y         
        
        bottom_tile_rt_x = top_tile_rt_x
        bottom_tile_rt_y = top_tile_rb_y 
        
        bottom_tile_lb_x = lb_x 
        bottom_tile_lb_y = lb_y
        
        bottom_tile_rb_x = rb_x 
        bottom_tile_rb_y = rb_y 

        bottom_tile_b_x = (bottom_tile_lt_x + bottom_tile_rt_x)/2.0
        bottom_tile_b_y = (bottom_tile_lt_y + bottom_tile_lb_y)/2.0
        bottom_tile_b_h = bottom_tile_lb_y - bottom_tile_lt_y
        bottom_tile_b_w = bottom_tile_rt_x - bottom_tile_lt_x        
        
        return [(top_tile_b_x, top_tile_b_y, top_tile_b_h, top_tile_b_w, lt_tile_num), (bottom_tile_b_x, bottom_tile_b_y, bottom_tile_b_h, bottom_tile_b_w, lb_tile_num)]

    elif lt_tile_num==lb_tile_num:
        # For left tile.
        left_tile_lt_x = lt_x 
        left_tile_lt_y = lt_y         
        
        left_tile_rt_x = lt_tile_r_x
        left_tile_rt_y = left_tile_lt_y 
        
        left_tile_lb_x = left_tile_lt_x 
        left_tile_lb_y = lb_y
        
        left_tile_rb_x = left_tile_rt_x 
        left_tile_rb_y = left_tile_lb_y
        
        left_tile_b_x = (left_tile_lt_x + left_tile_rt_x)/2.0
        left_tile_b_y = (left_tile_lt_y + left_tile_lb_y)/2.0
        left_tile_b_h = left_tile_lb_y - left_tile_lt_y
        left_tile_b_w = left_tile_rt_x - left_tile_lt_x
        

        # For right tile.
        right_tile_lt_x = left_tile_rt_x 
        right_tile_lt_y = left_tile_lt_y         
        
        right_tile_rt_x = rt_x
        right_tile_rt_y = right_tile_lt_y 
        
        right_tile_lb_x = left_tile_rb_x 
        right_tile_lb_y = left_tile_lb_y
        
        right_tile_rb_x = right_tile_rt_x 
        right_tile_rb_y = right_tile_lb_y 

        right_tile_b_x = (right_tile_lt_x + right_tile_rt_x)/2.0
        right_tile_b_y = (right_tile_lt_y + right_tile_lb_y)/2.0
        right_tile_b_h = right_tile_lb_y - right_tile_lt_y
        right_tile_b_w = right_tile_rt_x - right_tile_lt_x        
        
        return [(left_tile_b_x, left_tile_b_y, left_tile_b_h, left_tile_b_w, lt_tile_num), (right_tile_b_x, right_tile_b_y, right_tile_b_h, right_tile_b_w, rt_tile_num)]

    else:
        # For left_top tile.
        left_top_tile_lt_x = lt_x 
        left_top_tile_lt_y = lt_y
        
        
        left_top_tile_rt_x = lt_tile_r_x
        left_top_tile_rt_y = left_top_tile_lt_y 
        
        left_top_tile_lb_x = left_top_tile_lt_x 
        left_top_tile_lb_y = lt_tile_b_y
        
        left_top_tile_rb_x = left_top_tile_rt_x 
        left_top_tile_rb_y = left_top_tile_lb_y
        
        left_top_tile_b_x = (left_top_tile_lt_x + left_top_tile_rt_x)/2.0
        left_top_tile_b_y = (left_top_tile_lt_y + left_top_tile_lb_y)/2.0
        left_top_tile_b_h = left_top_tile_lb_y - left_top_tile_lt_y
        left_top_tile_b_w = left_top_tile_rt_x - left_top_tile_lt_x
        
        # For left_bottom tile.
        left_bottom_tile_lt_x = left_top_tile_lt_x 
        left_bottom_tile_lt_y = left_top_tile_lb_y
            
        left_bottom_tile_rt_x = left_top_tile_rb_x
        left_bottom_tile_rt_y = left_bottom_tile_lt_y 
        
        left_bottom_tile_lb_x = lb_x 
        left_bottom_tile_lb_y = lb_y
        
        left_bottom_tile_rb_x = left_bottom_tile_rt_x 
        left_bottom_tile_rb_y = left_bottom_tile_lb_y
        
        left_bottom_tile_b_x = (left_bottom_tile_lt_x + left_bottom_tile_rt_x)/2.0
        left_bottom_tile_b_y = (left_bottom_tile_lt_y + left_bottom_tile_lb_y)/2.0
        left_bottom_tile_b_h = left_bottom_tile_lb_y - left_bottom_tile_lt_y
        left_bottom_tile_b_w = left_bottom_tile_rt_x - left_bottom_tile_lt_x        
        

        # For right_top tile.
        right_top_tile_lt_x = left_top_tile_rt_x 
        right_top_tile_lt_y = left_top_tile_lt_y         
        
        right_top_tile_rt_x = rt_x
        right_top_tile_rt_y = right_top_tile_lt_y 
        
        right_top_tile_lb_x = left_top_tile_rb_x 
        right_top_tile_lb_y = left_top_tile_rb_y
        
        right_top_tile_rb_x = right_top_tile_rt_x 
        right_top_tile_rb_y = right_top_tile_lb_y 

        right_top_tile_b_x = (right_top_tile_lt_x + right_top_tile_rt_x)/2.0
        right_top_tile_b_y = (right_top_tile_lt_y + right_top_tile_lb_y)/2.0
        right_top_tile_b_h = right_top_tile_lb_y - right_top_tile_lt_y
        right_top_tile_b_w = right_top_tile_rt_x - right_top_tile_lt_x 

        # For right_bottom tile.
        right_bottom_tile_lt_x = right_top_tile_lb_x 
        right_bottom_tile_lt_y = right_top_tile_lb_y         
        
        right_bottom_tile_rt_x = right_top_tile_rt_x
        right_bottom_tile_rt_y = right_bottom_tile_lt_y 
        
        right_bottom_tile_lb_x = right_bottom_tile_lt_x 
        right_bottom_tile_lb_y = left_bottom_tile_rb_y
        
        right_bottom_tile_rb_x = rb_x 
        right_bottom_tile_rb_y = right_bottom_tile_lb_y 

        right_bottom_tile_b_x = (right_bottom_tile_lt_x + right_bottom_tile_rt_x)/2.0
        right_bottom_tile_b_y = (right_bottom_tile_lt_y + right_bottom_tile_lb_y)/2.0
        right_bottom_tile_b_h = right_bottom_tile_lb_y - right_bottom_tile_lt_y
        right_bottom_tile_b_w = right_bottom_tile_rt_x - right_bottom_tile_lt_x               
    
        return [(left_top_tile_b_x, left_top_tile_b_y, left_top_tile_b_h, left_top_tile_b_w, lt_tile_num), (left_bottom_tile_b_x, left_bottom_tile_b_y, left_bottom_tile_b_h, left_bottom_tile_b_w, lb_tile_num), (right_top_tile_b_x, right_top_tile_b_y, right_top_tile_b_h, right_top_tile_b_w, rt_tile_num), (right_bottom_tile_b_x, right_bottom_tile_b_y, right_bottom_tile_b_h, right_bottom_tile_b_w, rb_tile_num)]        

        
      






 