"""
Script to predict synapses in stormtiff images using previously trained models.

Swapnil 03/24
"""

import numpy as np
import glob
import math
import pickle
import os
from tifffile import tifffile
import multiprocessing
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dropout
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.optim as optim

import models
from get_data import getData

# import storm_analysis.sa_library.sa_h5py as saH5Py

# from cnn_input import cnnInput
# from make_tiles import makeTiles
# from locs_from_tiles_multiprocessing import locsFromTilesMultiprocessing
# from locs_per_storm_image import locsPerStormImage
# from locs_per_tile_3 import locsPerTile

# Limiting GPU memory growth. Source: https://www.tensorflow.org/guide/gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    
# Are you analyzing training data?
train = True
# train = False                                                                                                                                                                                                                                                                                                                                                             

# Get testing data.
# Set path to data files.
expfolder = "C:\\Users\\Swapnil\\Research\\synapse_segmentation\\"
data_directory = expfolder + "make_data\\"
testing_data_directory = data_directory + "testing_data\\"
# storm_exp_name = "750storm"
storm_exp_name = "647storm"
storm_exp_directory = testing_data_directory + storm_exp_name + "\\"
experiment_directory = expfolder + "experiments\\"


# Specify the tile-size of storm image section for training data.
tile_size = 400

# Get old tile size.
old_tile_size = 72


# Give the experiment name and experiment model name as a string.
exp_name = "experiment_1"

exp_mod = "experiment1_model"

# Set path for prediction files.
if train:
    if not os.path.exists(experiment_directory + "model_predictions_training_data\\"):
        os.mkdir(experiment_directory + "model_predictions_training_data\\")
        
    if not os.path.exists(experiment_directory + "model_predictions_training_data\\" + exp_name + "\\"):
        os.mkdir(experiment_directory + "model_predictions_training_data\\" + exp_name + "\\")    
        
    prediction_directory = experiment_directory + "model_predictions_training_data\\" + exp_name + "\\"
    
else:
    if not os.path.exists(experiment_directory + "model_predictions\\"):
        os.mkdir(experiment_directory + "model_predictions\\")
        
    if not os.path.exists(experiment_directory + "model_predictions\\" + exp_name + "\\"):
        os.mkdir(experiment_directory + "model_predictions\\" + exp_name + "\\")    
        
    prediction_directory = experiment_directory + "model_predictions\\" + exp_name + "\\"

# Get the path of the previously trained model for this experiment.
# model_directory = experiment_directory + "saved_models_copy\\"
model_directory = experiment_directory + "saved_models\\"
model_file = model_directory + exp_mod

# Get number of epochs for which the model was trained.
epochs = 1000

# input_data_directory = storm_exp_directory + "input_data_tiles_size_{}_copy2\\" .format(tile_size)
# output_data_directory = storm_exp_directory + "output_data_tiles_size_{}_copy2\\" .format(tile_size)

if train: 
    input_data_directory = storm_exp_directory + "input_data_tiles_size_{}_train_copy2\\" .format(tile_size)
    output_data_directory = storm_exp_directory + "output_data_tiles_size_{}_train_copy2\\" .format(tile_size)    
else: 
    input_data_directory = storm_exp_directory + "input_data_tiles_size_{}_copy2\\" .format(tile_size)
    output_data_directory = storm_exp_directory + "output_data_tiles_size_{}_copy2\\" .format(tile_size)  

# Get the size of data.

if train: 
    # data_size = 20
    data_size = 20000
else:
    data_size = 1000    

if os.path.exists(input_data_directory): tiles, tile_id_list = getData(input_data_directory, data_type="input", data_size=data_size)
    
else: print("Tiles data for this tile-size, tile-step and storm image section does not exist.")

if os.path.exists(output_data_directory): out_tiles, out_tile_id_list = getData(output_data_directory, data_type="output", data_size=data_size)

else: print("Locs data for this tile-size, tile-step and storm image section does not exist.")

# total number of input tiles for training
num_tiles = len(tiles)

# total number of locs files for training
num_out_tiles = len(out_tiles)
    
print("total number input tiles created for testing are {}\n" .format(num_tiles))
print("total number of output files created for testing are {}\n" .format(num_out_tiles))

# Convert tiles and locs lists to arrays.
tiles_test = np.array(tiles)
# tiles_train = np.array(tiles, dtype=object)
print("Training input from tiles has shape {}\n" .format(tiles_test.shape))

out_tiles_test = np.array(out_tiles)
print("Training output from files has shape {}\n" .format(out_tiles_test.shape))

# If necessary reshape input data from tiles to a shape the sequential model expects
tiles_test = np.reshape(tiles_test, (num_tiles, old_tile_size, old_tile_size, 1))     

# Reshape output data from locs to 1D array.
out_tiles_test = np.reshape(out_tiles_test, (num_tiles, 2, 1))

print("Testing input to model has shape {}\n" .format(tiles_test.shape))
print("Testing output from model has shape {}\n" .format(out_tiles_test.shape))

# Scale the pixel intensities to range [0,1]. 
tiles_test = tiles_test/255.0

# Loading the model back.
model = tf.keras.models.load_model(model_file)

# Initialize squared error
squared_error = 0.0

# Precision numbers initialization.
TP_ct = 0
TN_ct = 0
FP_ct = 0
FN_ct = 0

# for i in range(num_num_locs):
for i in range(len(tile_id_list)): 

    # # Use following lines for 1D flattened tiles.
    # tile_test = tiles_test[i,:]
    # tile_test = np.reshape(tile_test, (1, tile_size*tile_size))

    # Use following lines for 2D tiles.
    tile_test = tiles_test[i,:,:,:]
    tile_test = np.reshape(tile_test, (1, old_tile_size, old_tile_size, 1))
    
    # # Use following lines for resnet50 predicitions.
    # tile_test = tiles_test[i,:,:,:]
    # tile_test = np.reshape(tile_test, (1, tile_size, tile_size, 3))    

    out_tile_test = out_tiles_test[i,:,:]
    out_tile_test = np.reshape(out_tile_test, (1, 2, 1))
    
    # locs_pred_storm_file = prediction_directory + "storm_pred_locs_tile_" + str(i) + ".data"
    # num_locs_pred_tile_file = prediction_directory + "tile_pred_num_locs_tile_" + str(i+1) + ".data"
    out_pred_tile_file = prediction_directory + "tile_pred_out_tile_" + str(tile_id_list[i]) + ".data"    
    
    # Make predictions with trained model.
    pred = model.predict(tile_test)
    
    # Reshaping predictions.
    pred = np.reshape(pred, (1, 2, 1))    
    
    # Reshaping predictions.
    pred_test = np.reshape(pred, (1, 2))

    # Get precision numbers.
    if out_tile_test[0, 0, 0] == 1:
        if round(pred_test[0, 0]) == 1:
            TP_ct += 1
        else:
            FN_ct += 1
            
    else:
        if round(pred_test[0, 0]) == 1:
            FP_ct += 1
        else:
            TN_ct += 1
    
    
        
        
    
    # Add squared error
    squared_error += np.mean((pred_test.flatten() - out_tile_test.flatten())**2)

    # Writing the model predictions. 
    with open(out_pred_tile_file, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(pred_test, filehandle)      

# Compute mean-squared error in predicted localizations and print 
mean_squared_error = squared_error/num_out_tiles
print ("The mean-squared error in predicted localizations is: {}" .format(mean_squared_error))

print ("The TP count is is: {}" .format(TP_ct))
print ("The FP count is is: {}" .format(FP_ct))
print ("The TN count is is: {}" .format(TN_ct))
print ("The FN count is is: {}" .format(FN_ct))

print ("The precision is: {}" .format(TP_ct/(TP_ct+FP_ct)))
print ("The recall is: {}" .format(TP_ct/(TP_ct+FN_ct)))

#print("The mean-squared error in predicted localizations is: %.3f" %(mean_squared_error))

# # Save prediction error to file
# # f_out = open(prediction_directory + "prediction_error.csv","w")
# f_out = open(prediction_directory + "prediction_error.csv","a", newline='')
# f_out.write("prediction error after {} epochs is {}\n" .format(epochs, mean_squared_error))
# f_out.close()

# Save prediction error to file
with open(prediction_directory + "prediction_error.csv","a") as f_out:
    f_out.write("prediction error after {} epochs is {}\n" .format(epochs, mean_squared_error))
    


