"""
Script to test 'selective search' implementation in opencv python.  

See: https://learnopencv.com/selective-search-for-object-detection-cpp-python/

Swapnil 5/23
"""

import numpy as np
import sys
import cv2
import pickle

# print(cv2. __version__)

# Set the 'selective search' mode.
# selective_search_mode = None 
selective_search_mode = 'quality'
# selective_search_mode = 'fast'
 
 
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

tiles_directory = storm_exp_directory + "tiles_size_{}\\" .format(tile_size)

# Get tile number for testing tile.
t_num = 116
# t_num = 1

# Get image number for testing tile.
i_num = 5
# i_num = 3 

tile_file = tiles_directory + "tile_" + str(t_num) + "image_" + str(i_num) + ".data"

# Read from .data files. 
with open(tile_file, 'rb') as filehandle:
    tile = pickle.load(filehandle)

print(f"shape of tile is {tile.shape}.") 

# # If necessary reshape input data from tiles to a shape the opencv expects.
# tile = np.reshape(tile, (tile.shape[0], tile.shape[1], 3))

num_repeats = 3
tile = np.dstack([tile]*num_repeats)

print(f"shape of reshaped tile is {tile.shape}.")   

# speed-up using multithreads
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# read image
# im = cv2.imread('download_grey.jpg')
im = tile.copy()

# print(f'Data type of image is {type(im)}')
# print(f'Shape of image array is {im.shape}')

# create Selective Search Segmentation Object using default parameters
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# set input image on which we will run segmentation
ss.setBaseImage(im)
# ss.setBaseImage(tile)

# Switch to fast but low recall Selective Search method
if (selective_search_mode == 'fast'):
    ss.switchToSelectiveSearchFast()

# Switch to high recall but slow Selective Search method
elif (selective_search_mode == 'quality'):
    ss.switchToSelectiveSearchQuality()
# if argument is neither f nor q print help message
else:
    print("Please give valid 'selective search' mode.")
    sys.exit(1)

# run selective search segmentation on input image
rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))

# number of region proposals to show
numShowRects = 10
# increment to increase/decrease total number
# of reason proposals to be shown
increment = 10

while True:
    # create a copy of original image
    imOut = im.copy()

    # itereate over all the region proposals
    for i, rect in enumerate(rects):
        # draw rectangle for region proposal till numShowRects
        if (i < numShowRects):
            x, y, w, h = rect
            cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            break

    # show output
    cv2.imshow("Output", imOut)

    # record key press
    k = cv2.waitKey(0) & 0xFF

    # m is pressed
    if k == 109:
        # increase total number of rectangles to show by increment
        numShowRects += increment
    # l is pressed
    elif k == 108 and numShowRects > increment:
        # decrease total number of rectangles to show by increment
        numShowRects -= increment
    # q is pressed
    elif k == 113:
        break
# close image show window
cv2.destroyAllWindows()



# create a copy of original image
imOut = im.copy()

# itereate over all the region proposals
for i, rect in enumerate(rects):
    # draw rectangle for region proposal till numShowRects
    if (i < numShowRects):
        x, y, w, h = rect
        cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    else:
        break

# show output
cv2.imshow("Output", imOut)





