
import numpy as np
import pandas as pd 
import cv2
import dlib
from PIL import Image
from imageio import imread
from skimage.draw import circle

# File System & Utilities
import sys
import shutil
import os
import pathlib
from pathlib import Path
import glob
import time
import random
import math
from os.path import basename
from io import BytesIO

# Visualization & Display
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from google.colab import files
import IPython.display

# Computer Vision & Face Processing
import mediapipe as mp
from imutils import face_utils
import eos
import face_recognition
import faceBlendCommon as fbc

# Web Utilities
import requests
from bs4 import BeautifulSoup

# Deep Learning 
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical

# Keras Components
import h5py
from PIL import ImageFile
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K

# Path to the VGG16 weights file
VGG16_WEIGHTS_PATH = '/content/rcmalli_vggface_tf_notop_vgg16.h5'

# Data Preprocessing Pipeline

def locate_and_catalog_images(root_path: str) -> pd.DataFrame:
    """
    Recursively finds all JPG images in a directory and catalogs them.
    
    Args:
        root_path: Root directory to search for images
        
    Returns:
        DataFrame with class labels and file paths
    """
    image_catalog = []
    
    for root, directories, files in os.walk(root_path):
        for file in files:
            if ".jpg" in file:
                # Extract class from path and store with file path
                class_name = os.path.join(root, file).split("/")[-2]
                file_path = os.path.join(root, file)
                image_catalog.append((class_name, file_path))
                
    return pd.DataFrame(image_catalog, columns=['class', 'file_path'])

training_image_catalog = locate_and_catalog_images(train_images)

testing_image_catalog = locate_and_catalog_images(test_images)

for image_index in range(len(training_image_catalog)):
    transform(training_image_catalog['file_path'][image_index])

for image_index in range(len(testing_image_catalog)):
    transform(testing_image_catalog['file_path'][image_index])

def locate_and_catalog_processed_images(root_path: str) -> pd.DataFrame:
    """
    Recursively finds all processed JPG images in a directory and catalogs them.
    
    Args:
        root_path: Root directory to search for images
        
    Returns:
        DataFrame with class labels and file paths
    """
    processed_image_catalog = []
    
    for root, directories, files in os.walk(root_path):
        for file in files:
            if "_NEW_cropped.jpg" in file:
                # Extract class from path and store with file path
                class_name = os.path.join(root, file).split("/")[-2]
                file_path = os.path.join(root, file)
                processed_image_catalog.append((class_name, file_path))
                
    return pd.DataFrame(processed_image_catalog, columns=['class', 'file_path'])

# Re-catalog the processed training images
training_image_catalog = locate_and_catalog_processed_images(train_images)

# Re-catalog the processed testing images
testing_image_catalog = locate_and_catalog_processed_images(test_images)

# Training data generator with augmentation
training_data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True
)

# Testing data generator (no augmentation)
testing_data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Create training data flow from DataFrame
train_data_flow = training_data_generator.flow_from_dataframe(
    training_image_catalog, 
    x_col='file_path',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Create testing data flow from DataFrame
test_data_flow = testing_data_generator.flow_from_dataframe(
    testing_image_catalog, 
    x_col='file_path',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Model Architecture Definition

base_model = VGG16(
    input_shape=(224, 224, 3),  
    include_top=False,          
    weights=VGG16_WEIGHTS_PATH  
)

for layer in base_model.layers:
    layer.trainable = False

model_t1 = Sequential()

x = layers.Flatten()(base_model.output)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(5, activation='softmax')(x)


model_t1 = tf.keras.models.Model(base_model.input, x)


model_t1.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model_t1.summary()

# Model Training

ImageFile.LOAD_TRUNCATED_IMAGES = True

training_history = model_t1.fit(
    train_data_flow,              
    steps_per_epoch=train_data_flow.samples // batch_size,
    validation_data=test_data_flow,
    validation_steps=test_data_flow.samples // batch_size,
    epochs=50,
    callbacks=[es, chkpt]
)

# Save the trained model

model_t1.save('face_shape_classifier.h5')
model_t1.save_weights('face_shape_classifier_weights.h5')

# Make predictions with the model

model_t1.predict(
    np.asarray(crop_image).reshape(
        -1, 
        np.asarray(crop_image).shape[0], 
        np.asarray(crop_image).shape[1], 
        np.asarray(crop_image).shape[2]
    )
)