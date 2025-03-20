
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import sys
import cv2
import dlib
from PIL import Image, ImageDraw
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


# FACIAL ANALYSIS FUNCTIONS

def distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        point_a: First point (x, y)
        point_b: Second point (x, y)
        
    Returns:
        Float distance between points
    """
    return np.sqrt(np.square(point_a[0] - point_b[0]) + np.square(point_a[1] - point_b[1]))


def scale_rotate_translate(
    image: np.ndarray,
    angle: float,
    center: Optional[Tuple[float, float]] = None,
    new_center: Optional[Tuple[float, float]] = None,
    scale: Optional[float] = None,
    resample: int = Image.BICUBIC
) -> Image.Image:
    """
    Apply geometric transformations to an image.
    
    Args:
        image: Source image
        angle: Rotation angle in radians
        center: Rotation center point
        new_center: New center point
        scale: Scaling factor
        resample: Resampling filter
        
    Returns:
        Transformed image
    """
    if (scale is None) and (center is None):
        return Image.fromarray(image, 'RGB').rotate(angle=angle, resample=resample)
    
    nx, ny = x, y = center
    sx = sy = 1.0
    
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    
    cosine = math.cos(angle)
    sine = math.sin(angle)
    
    a = cosine/sx
    b = sine/sx
    c = x-nx*a-ny*b
    d = -sine/sy
    e = cosine/sy
    f = y-nx*d-ny*e
    
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
    Crop and align a face based on eye positions.
    
    Args:
        image: Source image
        eye_left: Left eye position
        eye_right: Right eye position
        offset_pct: Margin percentage
        dest_sz: Output size
        
    Returns:
        Cropped and aligned face
    """
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])

    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))

    eye_dist = distance(eye_left, eye_right)

    reference = dest_sz[0] - 2.0 * offset_h

    scale = float(eye_dist) / float(reference)

    rotated_image = scale_rotate_translate(image, center=eye_left, angle=rotation)

    crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
    crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)

    cropped_image = rotated_image.crop((
        int(crop_xy[0]),
        int(crop_xy[1]),
        int(crop_xy[0] + crop_size[0]),
        int(crop_xy[1] + crop_size[1])
    ))

    return cropped_image.resize(dest_sz, Image.ANTIALIAS)


# MAIN EXECUTION


FACIAL_REGIONS = [
    'chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge',
    'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip'
]

IMAGE_PATH = '/content/output/testNoBackground.jpg'

facial_canvas = face_recognition.load_image_file(IMAGE_PATH)
facial_landmarks_map = face_recognition.face_landmarks(facial_canvas)

landmark_points = []

for face_structure in facial_landmarks_map:
    for feature_region in FACIAL_REGIONS:
        for point_location in face_structure[feature_region]:
            for pixel_coordinate in point_location:
                landmark_points.append(pixel_coordinate)

    left_eye_x, left_eye_y = landmark_points[72], landmark_points[73]
    right_eye_x, right_eye_y = landmark_points[90], landmark_points[91]

    eye_positions = []
    eye_positions.append(landmark_points[72:74])
    eye_positions.append(landmark_points[90:92])
    
    normalized_face = crop_face(
        facial_canvas,
        eye_left=(left_eye_x, left_eye_y),
        eye_right=(right_eye_x, right_eye_y),
        offset_pct=(0.34, 0.34),
        dest_sz=(224, 224)
    )

CROPPED_IMAGE_PATH = f"{IMAGE_PATH}_NEW_cropped.jpg"
normalized_face.save(CROPPED_IMAGE_PATH)

landmark_points = []
face_count = 0

cropped_facial_canvas = face_recognition.load_image_file(CROPPED_IMAGE_PATH)
cropped_facial_landmarks = face_recognition.face_landmarks(cropped_facial_canvas)
n
visualization_canvas = Image.fromarray(cropped_facial_canvas)
drawing_context = ImageDraw.Draw(visualization_canvas)

for face_structure in cropped_facial_landmarks:
    for feature_region in face_structure.keys():
        drawing_context.line(face_structure[feature_region], width=5)
        drawing_context.point(face_structure[feature_region], fill=(255, 255, 255))

    for feature_region in FACIAL_REGIONS:
        for point_location in face_structure[feature_region]:
            for pixel_coordinate in point_location:
                landmark_points.append(pixel_coordinate)


# FACIAL MEASUREMENTS & PROPORTIONS═

face_width = np.sqrt(
    np.square(landmark_points[0] - landmark_points[32]) + 
    np.square(landmark_points[1] - landmark_points[33])
)

face_height = np.sqrt(
    np.square(landmark_points[16] - landmark_points[56]) + 
    np.square(landmark_points[17] - landmark_points[57])
) * 2

height_to_width_ratio = face_height / face_width

jaw_width = np.sqrt(
    np.square(landmark_points[12] - landmark_points[20]) + 
    np.square(landmark_points[13] - landmark_points[21])
)

jaw_to_face_width_ratio = jaw_width / face_width

eyebrow_arch_height = (
    abs(landmark_points[43] - landmark_points[39]) + 
    abs(landmark_points[53] - landmark_points[49])
) / 2

lip_width = abs(landmark_points[139] - landmark_points[127])

eye_height = (
    abs(landmark_points[75] - landmark_points[81]) + 
    abs(landmark_points[87] - landmark_points[95])
) / 2

eye_width = (
    abs(landmark_points[72] - landmark_points[78]) + 
    abs(landmark_points[84] - landmark_points[90])
) / 2

eye_height_to_width_ratio = eye_height / eye_width

nose_length = abs(landmark_points[57] - landmark_points[67])

nose_width = abs(landmark_points[70] - landmark_points[62])