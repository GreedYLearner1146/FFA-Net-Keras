# Import the packages.

# Importing the library.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
#import keras
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import cv2
import glob
from os import listdir
from numpy import asarray
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import re
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ReLU, Add, MaxPool2D, UpSampling2D, BatchNormalization, concatenate, Subtract
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Add, Activation, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.activations import relu

############## FFA-Net architecture attempt in Keras #########################
# Basic Block Structure. 3 of them.

def BasicBlockStructure(input_bbs):

  conv1 = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(input_bbs)
  Add1 = keras.layers.Add()([input_bbs,conv1])
  conv2 = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', use_bias = True)(input_bbs)

  # Channel Attention
  avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(1, 1),strides=(1, 1), padding='valid')(conv2)  # Incldue average Pool 2D
  conv3 = Conv2D(filters = 32, kernel_size = 1, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(avg_pool_2d)
  conv4= Conv2D(filters = 32, kernel_size = 1, strides = 1, padding = 'same', use_bias = True, activation = 'sigmoid')(conv3)
  MultiplyI = keras.layers.Multiply()([conv2,conv4])  # Multiply skip connection.

  # Spatial Attention
  conv5 = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(MultiplyI)
  conv6 = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'sigmoid')(conv5)
  MultiplyII = keras.layers.Multiply()([MultiplyI,conv6])  # Multiply skip connection.

  Add2 = keras.layers.Add()([input_bbs,MultiplyII])

  return Add2

# Group Structure of 19 basic block structure.

def GroupStructure(input_gs):

    BBS1 = BasicBlockStructure(input_gs)
    BBS2 = BasicBlockStructure(BBS1)
    BBS3 = BasicBlockStructure(BBS2)
    BBS4 = BasicBlockStructure(BBS3)
    BBS5 = BasicBlockStructure(BBS4)
    BBS6 = BasicBlockStructure(BBS5)
    BBS7 = BasicBlockStructure(BBS6)
    BBS8 = BasicBlockStructure(BBS7)
    BBS9 = BasicBlockStructure(BBS8)
    BBS10 = BasicBlockStructure(BBS9)
    BBS11 = BasicBlockStructure(BBS10)
    BBS12 = BasicBlockStructure(BBS11)
    BBS13 = BasicBlockStructure(BBS12)
    BBS14 = BasicBlockStructure(BBS13)
    BBS15 = BasicBlockStructure(BBS14)
    BBS16 = BasicBlockStructure(BBS15)
    BBS17 = BasicBlockStructure(BBS16)
    BBS18 = BasicBlockStructure(BBS17)
    BBS19 = BasicBlockStructure(BBS18)

    convGS = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', use_bias = True)(BBS19)
    AddGS = keras.layers.Add()([input_gs,convGS])

    return AddGS

# Main FFA-Net with 19 Basic Block Structure making up 1 Group stucture.

def FFANet():

    in_image = Input(shape = (256,256,3))
    convI = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', use_bias = True)(in_image)

    GS1 = GroupStructure(convI)  # Group structure 1.
    GS2 = GroupStructure(GS1)  # Group structure 2.
    GS3 = GroupStructure(GS2)  # Group structure 3.

    concatI = tf.keras.ops.concatenate([GS1,GS2,GS3], axis = -1)  # Concatenate.

    # Channel Attention
    avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(1, 1),strides=(1, 1), padding='valid')(concatI)  # Incldue average Pool 2D
    convII = Conv2D(filters = 32, kernel_size = 1, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(avg_pool_2d)
    convIII= Conv2D(filters = 32, kernel_size = 1, strides = 1, padding = 'same', use_bias = True, activation = 'sigmoid')(convII)
    MultiplyI = keras.layers.Multiply()([convII,convIII])  # Multiply skip connection.

    # Spatial Attention
    convIV = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(MultiplyI)
    convV = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'sigmoid')(convIV)
    MultiplyII = keras.layers.Multiply()([MultiplyI,convV])  # Multiply skip connection.

    # Two convolutional layer.

    convVI = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', use_bias = True)(MultiplyII)
    convVII = Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = 'same', use_bias = True)(convVI)

    # Final skip connection layer.

    AddFFA = keras.layers.Add()([in_image,convVII])

    # Model definition

    ffamodel = Model(in_image, AddFFA)

    return ffamodel

FFANet = FFANet()  # Total 1,763,203 parameters.
FFANet.summary()
