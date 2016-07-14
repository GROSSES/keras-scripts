from keras.layers.convolutional import ZeroPadding2D,MaxPooling2D,Convolution2D
from keras.layers.core import Dense,Dropout,Flatten
from keras.models import Sequential
import h5py
import numpy as np

def get_vgg_face(): 
    # build the VGG16 network
    f = h5py.File('vgg_face_16.h5')
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224,224)))
    
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1',weights=[f['conv1_1_weight'][:],
                                                                                 f['conv1_1_bias'][:]]))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2',weights=[f['conv1_2_weight'][:],
                                                                                 f['conv1_2_bias'][:]]))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1',weights=[f['conv2_1_weight'][:],
                                                                                 f['conv2_1_bias'][:]]))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2',weights=[f['conv2_2_weight'][:],
                                                                                 f['conv2_2_bias'][:]]))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1',weights=[f['conv3_1_weight'][:],
                                                                                 f['conv3_1_bias'][:]]))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2',weights=[f['conv3_2_weight'][:],
                                                                                 f['conv3_2_bias'][:]]))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3',weights=[f['conv3_3_weight'][:],
                                                                                 f['conv3_3_bias'][:]]))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1',weights=[f['conv4_1_weight'][:],
                                                                                 f['conv4_1_bias'][:]]))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2',weights=[f['conv4_2_weight'][:],
                                                                                 f['conv4_2_bias'][:]]))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3',weights=[f['conv4_3_weight'][:],
                                                                                 f['conv4_3_bias'][:]]))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1',weights=[f['conv5_1_weight'][:],
                                                                                 f['conv5_1_bias'][:]]))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2',weights=[f['conv5_2_weight'][:],
                                                                                 f['conv5_2_bias'][:]]))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3',weights=[f['conv5_3_weight'][:],
                                                                                 f['conv5_3_bias'][:]]))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu',weights=[np.transpose(f['fc6_weight'][:]),f['fc6_bias'][:]]))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu',weights=[np.transpose(f['fc7_weight'][:]),f['fc7_bias'][:]]))
    model.add(Dropout(0.5))
    model.add(Dense(2622, activation='softmax',weights=[np.transpose(f['fc8_weight'][:]),f['fc8_bias'][:]]))

    f.close()
    return model
