
# » Data Science & Image Processing
import numpy as np
import pandas as pd
import cv2
import dlib
from PIL import Image
from PIL import ImageFile
from imageio import imread
from skimage.draw import circle

# » File System & Utilities
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

# » Visualization & Display
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from google.colab import files
import IPython.display

# » Computer Vision & Face Processing
import mediapipe as mp
from imutils import face_utils
import eos
import face_recognition
import faceBlendCommon as fbc

# » Web Utilities
import requests
from bs4 import BeautifulSoup

# » Deep Learning
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical

# » Keras Components
import h5py
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K


# 🔽 MODNet SETUP

def setup_modnet() -> None:
    """
    Clone the MODNet repository and download pre-trained weights.
    
    This function changes to the /content directory, clones the MODNet repository
    if it doesn't exist, and downloads the pre-trained model checkpoint.
    """
    %cd /content
    
    if not os.path.exists('MODNet'):
        !git clone https://github.com/ZHKKKe/MODNet
    
    %cd MODNet/
    
    pretrained_model_path = 'pretrained/modnet_photographic_portrait_matting.ckpt'
    
    if not os.path.exists(pretrained_model_path):
        !gdown --id 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz \
                -O pretrained/modnet_photographic_portrait_matting.ckpt


def prepare_directories() -> None:
    """
    Create input and output directories for image processing.
    
    This function creates (or cleans) the input and output directories,
    and moves the test image to the input directory.
    """

    input_directory = '/content/input'
    if os.path.exists(input_directory):
        shutil.rmtree(input_directory)
    os.makedirs(input_directory)

    output_directory = '/content/output'
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    shutil.move('/content/test.jpg', os.path.join(input_directory, 'test.jpg'))


def run_modnet_inference() -> None:
    """
    Execute MODNet inference to remove the image background.
    
    This function runs the MODNet inference script on the input image
    to generate a matte that separates the foreground from the background.
    """
    !python -m demo.image_matting.colab.inference \
            --input-path /content/input/ \
            --output-path /content/output/ \
            --ckpt-path ./pretrained/modnet_photographic_portrait_matting.ckpt


def process_output(image_name: str = 'test.jpg') -> None:
    """
    Process the MODNet output to create a foreground-only image.
    
    Args:
        image_name: Name of the input image file
        
    This function combines the original image with the generated matte
    to create a new image with the background removed.
    """
    matte_name = image_name.split('.')[0] + '.png'
    
    original_image = Image.open(os.path.join('/content/input', image_name))
    alpha_matte = Image.open(os.path.join('/content/output', matte_name))
    
    width, height = original_image.width, original_image.height
    resized_width, resized_height = 800, int(height * 800 / (3 * width))
    
    image_array = np.asarray(original_image)

    if len(image_array.shape) == 2:
        image_array = image_array[:, :, None]
    if image_array.shape[2] == 1:
        image_array = np.repeat(image_array, 3, axis=2)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, 0:3]
    
    matte_array = np.repeat(np.asarray(alpha_matte)[:, :, None], 3, axis=2) / 255
    foreground_array = image_array * matte_array + np.full(image_array.shape, 255) * (1 - matte_array)
    
    combined_array = np.concatenate((image_array, foreground_array, matte_array * 255), axis=1)
    combined_image = Image.fromarray(np.uint8(combined_array)).resize((resized_width, resized_height))

    foreground_image = Image.fromarray(np.uint8(foreground_array)).resize((width, height))
    
    foreground_image.save("/content/output/testNoBackground.jpg")



# MAIN EXECUTION

setup_modnet()

prepare_directories()

run_modnet_inference()

process_output('test.jpg')