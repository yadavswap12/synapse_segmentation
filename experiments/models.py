"""
This script is a collection of neural network models built using sequential() module
in keras. It allows to choose broad architecture of the network as well as set some of 
the hyperparameters of the network.  
"""

import numpy as np
import math
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.optim as optim

class cnnRegression(object):
    
    """
    Class for CNN model designed for regression task.

    This has cnnBuild(), fcBuild functions to modify the architecture and parameters of the network.
    """
    
    def __init__(self, parameters, **kwds):
    
        """
        This function initializes the network.

        parameters- A tupple containing following parameters.
        Eg. paramaters = (conv_input_shape, output_nodes, conv_layers, fc_layers)
            
        conv_input_shape - A tupple which gives shape of input (set of images) to the network.
        Eg. conv_input_shape = (num_image, height, width, channels)
            
        output_nodes - Number of output nodes of network for regression.
        conv_layers - Number of Conv2D layers.
        fc_layers - Number of fully-connected (Dense) layers after Conv2D layers.                
        """
        
        # Get the structure of the network.
        self.conv_input_shape, self.output_nodes, self.conv_layers, self.fc_layers = parameters
        self.num_image, self.height, self.width, self.channels = self.conv_input_shape
        
    def cnnBuild(self, conv_stride, conv_pad, conv_activ, conv_reg, conv_init, conv_drop, conv_BN, pool_size, pool_stride, pool_pad):
    
        """
        This function builds network with given parameters.
        
        conv_stride - A tupple giving strides of the convolution window along horizontal and vertical direction.
        Eg. conv_stride = (1,1).

        conv_pad - A string giving the type of padding for convolution process.
        Eg. conv_pad = "same" or  "valid".
            
        conv_activ - A string giving the type of activation function for Conv2D layer.
        Eg. conv_activ = "relu" or  "tanh".            
                
        conv_reg - The regularization method to be used for Conv2D layers.
        If no regularization is to be used then set conv_reg = None.

        conv_init - The kernel initialization method to be used for Conv2D layer weights.
        If default initialization (glorot_uniform) is to be used then set conv_init = "glorot_uniform".

        conv_drop - The value of the dropout probability for dropout method in Conv2D layers.
        If no dropout is to be used then set conv_drop = "None".
        
        conv_BN - A boolean variable indicating whether or not batch normalization is to be implemented in Conv2D layers.
        Eg. If conv_BN = False then don't implement batch normalization.
        
        pool_size - window size of MaxPooling2D().
        Eg. pool_size = (2,2).
        
        pool_stride - A tupple giving strides of the pooling window along horizontal and vertical direction.
        Eg. pool_stride = 2.

        pool_pad - A string giving the type of padding for pooling process.
        Eg. pool_pad = "valid".
        """
        
        # Assign number of filters for each Conv2D layer.
        # Start with 64 filters for first layer and then double the filters for additional layer.
        filters = []
        
        # Shape of the input to the first Conv2D layer.
        ip_shape = (self.conv_input_shape[1],self.conv_input_shape[2],self.conv_input_shape[3])
        
        for conv_layer in range(self.conv_layers):
            
            f = 2**(5+conv_layer)
            filters.append(f)
            
        # Choose kernel size according to input image size.
        if ((self.height > 512) | (self.width > 512)):
            
            ker_size = 7
            
        elif ((self.height > 128) | (self.width > 128)):     
    
            ker_size = 5
            
        else:

            ker_size = 3            
    
        # create cnn model
        cnnReg = Sequential()
        
        for conv_layer in range(self.conv_layers):
            
            # For creating first convolutional layer, specify input shape.
            if (conv_layer == 0):
            
                cnnReg.add(Conv2D(filters[conv_layer], kernel_size=ker_size, strides=conv_stride, padding=conv_pad, activation=conv_activ, kernel_initializer=conv_init, kernel_regularizer=conv_reg, input_shape=ip_shape))
                cnnReg.add(MaxPooling2D(pool_size=pool_size, strides=pool_stride, padding=pool_pad))
                
                if conv_BN: cnnReg.add(BatchNormalization(axis=-1))

                if conv_drop: cnnReg.add(Dropout(conv_drop))            
                
            
            else:
            
                # Add rest of the layers.
                cnnReg.add(Conv2D(filters[conv_layer], kernel_size=ker_size, strides=conv_stride, padding=conv_pad, activation=conv_activ, kernel_initializer=conv_init, kernel_regularizer=conv_reg))
                cnnReg.add(MaxPooling2D(pool_size=pool_size, strides=pool_stride, padding=pool_pad))
                
                if conv_BN: cnnReg.add(BatchNormalization(axis=-1))

                if conv_drop: cnnReg.add(Dropout(conv_drop))                            
        
        return cnnReg

        
    def fcBuild(self, cnnReg, fc_layers, fc_activ, fc_reg, fc_init, fc_drop, fc_BN, fc_output_layer_units):
    
        """
        This function builds fc network with given parameters at the end of given CNN network.
        
        cnnReg - CNN model built with Sequential() at the end of which fully-connected network is to be added.
        
        fc_layers - It is a list of number of nodes in each fully-connected layer.
        Eg. fc_layers = [3,4,6,3] means that there are four hidden fully-connected layers.
        The first hidden layer has 3 nodes, the second hidden layer has 4 nodes and so on.
        if fc_layers = [] (empty list) then use predefined list fc_layers_custom.

        fc_activ - A string giving the type of activation function for fully-connected layer.
        Eg. fc_activ = "relu" or  "tanh". 
        
        fc_reg - The regularization method to be used for fully-connected layers.
        If no regularization is to be used then set fc_reg = None.
        
        fc_init - The initialization method to be used for fully-connected layer weights.
        If default initialization (glorot_uniform) is to be used then set conv_init = "glorot_uniform". 
        
        fc_drop - The value of the dropout probability for dropout method in fully-connected layers.
        If no dropout is to be used then set fc_drop = None. 
        
        fc_BN - A boolean variable indicating whether or not batch normalization is to be implemented in fully-connected layers.
        Eg. If fc_BN = False then don't implement batch normalization.
        
        fc_output_layer_units - Units in the output layer of fully-connected network.
        """
        
        # Define custom fully-connected layers list.
        fc_layers_custom = [4096, 4096]
        
        # flattening the output of conv2d layers. May have to add batch normalization(BN) layer.
        cnnReg.add(Flatten())

        if fc_layers:
        
            for fc_layer in range(len(fc_layers)):
            
                # adding dense and dropout layers
                cnnReg.add(Dense(units = fc_layers[fc_layer], activation=fc_activ, kernel_initializer=fc_init, kernel_regularizer=fc_reg))
                
                if fc_BN: cnnReg.add(BatchNormalization(axis=-1))                
                
                if fc_drop: cnnReg.add(Dropout(fc_drop))            
            
        else:

            for fc_layer in range(len(fc_layers_custom)):
            
                # adding dense and dropout layers
                cnnReg.add(Dense(units = fc_layers_custom[fc_layer], activation=fc_activ, kernel_initializer=fc_init, kernel_regularizer=fc_reg))
                
                if fc_BN: cnnReg.add(BatchNormalization(axis=-1))                                
                
                if fc_drop: cnnReg.add(Dropout(fc_drop)) 
         
        # Adding the output layer
        cnnReg.add(Dense(units = fc_output_layer_units, activation="linear"))

        return cnnReg
        
class nnRegression(object):

    """
    Class for a fully-connected NN model designed for regression task.

    This has nnBuild() function to modify the architecture and parameters of the network.
    """
    
    def __init__(self, parameters, **kwds):
    
        """
        This function initializes the network.

        parameters- A tuple containing following parameters.
        Eg. parameters = (nn_input_shape, output_nodes, nn_layers)
            
        nn_input_shape - A tuple which gives shape of input to the network.
        Eg. nn_input_shape = (num_example, input_nodes)
            
        output_nodes - Number of output nodes of network for regression.
        nn_layers - It is a list of number of nodes in each fully-connected layer.
        Eg. nn_layers = [3,4,6,3] means that there are four hidden fully-connected layers.
        The first hidden layer has 3 nodes, the second hidden layer has 4 nodes and so on.
        """
        
        # Get the structure of the network.
        self.nn_input_shape, self.output_nodes, self.nn_layers = parameters
        self.num_example, self.input_nodes = self.nn_input_shape

    def nnBuild(self, nn_activ, nn_reg, nn_init, nn_drop, nn_BN):
    
        """
        This function builds network with given parameters.
            
        nn_activ - A string giving the type of activation function for fully-connected layer.
        Eg. nn_activ = "relu" or  "tanh".            
                
        nn_reg - The regularization method to be used for fully-connected layers.
        If no regularization is to be used then set nn_reg = "None".

        nn_init - The initialization method to be used for Conv2D layer weights.
        If no initialization is to be used then set conv_init = "None".

        nn_drop - The value of the dropout probability for dropout method in fully-connected layers.
        If no dropout is to be used then set nn_drop = "None".
        
        nn_BN - A boolean variable indicating whether or not batch normalization is to be implemented in fully-connected layers.
        Eg. If nn_BN = False then don't implement batch normalization.
        """
    
        # create nn model
        nnReg = Sequential()
        
        for nn_layer in range(len(self.nn_layers)):
            
            # adding dense and dropout layers
            nnReg.add(Dense(units = self.nn_layers[nn_layer], activation=nn_activ))
            
            if nn_BN: nnReg.add(BatchNormalization(axis=-1))                
            
            if nn_drop: nnReg.add(Dropout(nn_drop))
            
        # Adding the output layer
        nnReg.add(Dense(units = self.output_nodes, activation="linear"))    
        
        return nnReg        
        
                
                

                
                    
                
                    
                
                
                    
                    
    



    
