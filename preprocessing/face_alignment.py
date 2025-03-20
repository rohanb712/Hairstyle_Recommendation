
import numpy as np
import pandas as pd
import sys
import cv2
import dlib
import numpy as np
from PIL import Image
import shutil
from google.colab.patches import cv2_imshow
from google.colab import files
import mediapipe as mp
from imutils import face_utils
import eos
from imageio import imread
from io import BytesIO
import IPython.display
import requests
from bs4 import BeautifulSoup
import time
from PIL import Image, ImageDraw
import face_recognition
from os.path import basename
import math
import pathlib
from pathlib import Path
import os
import random
import matplotlib.pyplot as plt
from skimage.draw import circle
import glob
import h5py
from PIL import ImageFile
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras import backend as K
import faceBlendCommon as fbc
from tensorflow.keras import layers
import tensorflow as tf
from typing import Tuple, List, Dict, Any, Optional, Union, Callable


def transform(image_path: str) -> None:
    """
    FACIAL LANDMARK EXTRACTION AND ALIGNMENT
    
    This function extracts facial landmarks from an image and produces
    a standardized, aligned face crop.
    
    Args:
        image_path: Path to the source image file
        
    Returns:
        None (saves cropped image to disk with "_NEW_cropped.jpg" suffix)
    """
    facial_canvas = face_recognition.load_image_file(image_path)
    facial_topography = face_recognition.face_landmarks(facial_canvas)
    
    # Define the regions of interest on the face
    facial_regions = [
        'chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge',
        'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip'
    ]
    
    # Flatten all landmarks into a single list of coordinates
    landmark_coordinates = []
    
    # Process each detected face
    for face_map in facial_topography:
        # Extract all landmark points from all facial regions
        for feature_zone in facial_regions:
            for point_location in face_map[feature_zone]:
                for pixel_coordinate in point_location:
                    landmark_coordinates.append(pixel_coordinate)
                    
        # Locate eye positions (crucial for alignment)
        # Indices 72-73: left eye corner, 90-91: right eye corner
        left_eye_x, left_eye_y = landmark_coordinates[72], landmark_coordinates[73]
        right_eye_x, right_eye_y = landmark_coordinates[90], landmark_coordinates[91]
        
        eye_anchors = []
        eye_anchors.append(landmark_coordinates[72:74])
        eye_anchors.append(landmark_coordinates[90:92])
        
        # Perform face alignment and cropping
        normalized_face = crop_face(
            facial_canvas, 
            eye_left=(left_eye_x, left_eye_y),
            eye_right=(right_eye_x, right_eye_y),
            offset_pct=(0.34, 0.34),
            dest_sz=(300, 300)
        )
        
        # Save the processed face
        normalized_face.save(f"{image_path}_NEW_cropped.jpg")


def euclidean_distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    """
    EUCLIDEAN DISTANCE CALCULATOR
    
    Calculates the Euclidean distance between two 2D points.
    
    Args:
        point_a: First point as (x, y) tuple
        point_b: Second point as (x, y) tuple
        
    Returns:
        Float distance between the points
    """
    delta_x = point_b[0] - point_a[0]
    delta_y = point_b[1] - point_a[1]
    
    return math.sqrt(delta_x * delta_x + delta_y * delta_y)


def scale_rotate_translate(
    image: np.ndarray,
    angle: float,
    center: Optional[Tuple[float, float]] = None,
    new_center: Optional[Tuple[float, float]] = None,
    scale: Optional[float] = None,
    resample: int = Image.BICUBIC
) -> Image.Image:
    """
    GEOMETRIC TRANSFORMATION ENGINE
    
    Applies scaling, rotation, and translation to an image.
    
    Args:
        image: Source image as numpy array
        angle: Rotation angle in radians
        center: Center of rotation (x, y) in source image
        new_center: Center position (x, y) in output image
        scale: Scaling factor
        resample: Resampling filter to use
        
    Returns:
        Transformed PIL Image
    """
    if (scale is None) and (center is None):
        return Image.fromarray(image, 'RGB').rotate(angle=angle, resample=resample)
    
    # Extract center coordinates
    nx, ny = x, y = center
    
    # Default scale factors
    sx = sy = 1.0
    
    # Apply new center if specified
    if new_center:
        (nx, ny) = new_center
        
    # Apply scaling if specified
    if scale:
        (sx, sy) = (scale, scale)
        
    # Precompute trigonometric values
    cosine = math.cos(angle)
    sine = math.sin(angle)
    
    # Calculate affine transformation matrix
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    
    # Apply transformation
    pil_image = Image.fromarray(image, 'RGB')
    return pil_image.transform(pil_image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)


def crop_face(
    image: np.ndarray,
    eye_left: Tuple[float, float] = (0, 0),
    eye_right: Tuple[float, float] = (0, 0),
    offset_pct: Tuple[float, float] = (0.3, 0.3),
    dest_sz: Tuple[int, int] = (600, 600)
) -> Image.Image:
    """
    FACIAL ALIGNMENT AND CROPPING
    
    Creates a standardized face crop centered around the eye positions.
    
    Args:
        image: Source image as numpy array
        eye_left: Left eye position (x, y)
        eye_right: Right eye position (x, y)
        offset_pct: Percentage of width/height to use as margin
        dest_sz: Output image dimensions (width, height)
        
    Returns:
        Aligned and cropped face as PIL Image
    """
    # Calculate margins in the output image
    horizontal_margin = math.floor(float(offset_pct[0]) * dest_sz[0])
    vertical_margin = math.floor(float(offset_pct[1]) * dest_sz[1])
    
    # Get eye direction vector
    eye_direction = (
        eye_right[0] - eye_left[0],
        eye_right[1] - eye_left[1]
    )
    
    # Calculate rotation angle in radians
    rotation_angle = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    
    # Calculate distance between eyes
    eye_distance = euclidean_distance(eye_left, eye_right)
    
    # Calculate reference eye width
    reference_eye_width = dest_sz[0] - 2.0 * horizontal_margin
    
    # Calculate scale factor
    scale_factor = float(eye_distance) / float(reference_eye_width)
    
    # Rotate image around the left eye
    aligned_image = scale_rotate_translate(
        image, 
        center=eye_left,
        angle=rotation_angle
    )
    
    # Calculate crop boundaries
    crop_origin = (
        eye_left[0] - scale_factor * horizontal_margin,
        eye_left[1] - scale_factor * vertical_margin
    )
    
    crop_dimensions = (
        dest_sz[0] * scale_factor,
        dest_sz[1] * scale_factor
    )
    
    # Crop the rotated image
    cropped_image = aligned_image.crop((
        int(crop_origin[0]),
        int(crop_origin[1]),
        int(crop_origin[0] + crop_dimensions[0]),
        int(crop_origin[1] + crop_dimensions[1])
    ))
    
    # Resize to final dimensions
    return cropped_image.resize(dest_sz, Image.ANTIALIAS)