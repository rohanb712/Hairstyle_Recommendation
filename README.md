# Hairstyle Recommendation System

## Overview

This AI-powered application helps users discover hairstyles that suit their face shape. The system analyzes facial features, determines face shape classification, and recommends appropriate hairstyles based on facial proportions and established styling guidelines.

## Features

- **Face Detection & Landmark Extraction**: Accurately identifies facial features
- **Background Removal**: Isolates the face from background elements
- **Face Shape Classification**: Analyzes facial measurements to determine face shape type
- **Personalized Recommendations**: Suggests hairstyles tailored to the user's face shape
- **Visual Guidance**: Provides visual examples of recommended styles

## Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- OpenCV
- dlib
- PIL/Pillow
- NumPy/Pandas
- face_recognition library

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hairstyle-recommendation-system.git
cd hairstyle-recommendation-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
```bash
./setup_resources.sh
```

## Project Structure

```
hairstyle-recommendation-system/
├── utils/                         # Core facial processing utilities
│   ├── facial_landmark_utils.py        # Facial landmark detection and manipulation
│   └── face_detection_model.xml        # XML file for face detection
├── classifier/                    # Face shape classification model
│   └── face_shape_classifier.py        # VGG16-based model for shape detection
├── segmentation/                  # Portrait background removal
│   └── portrait_background_remover.py  # MODNet implementation
├── preprocessing/                 # Face alignment and preprocessing
│   └── face_alignment.py               # Face alignment and standardization
├── input/                         # Directory for input images
├── notebook/                      # Notebooks for data exploration
│   └── hairstyle_data_analysis.ipynb   # EDA on hairstyle dataset
├── image_output/                  # Directory for processed images
├── setup_resources.sh             # Script to download required models
├── hairstyle_recommender.py       # Main application script
└── README.md                      # Project documentation
```

## Usage

### Basic Usage
```bash
python hairstyle_recommender.py --image path/to/your/image.jpg
```

### Options
```bash
python hairstyle_recommender.py --help

# Output:
# --image       Path to input image
# --save-steps  Save intermediate processing steps
# --top-n       Number of recommended hairstyles (default: 5)
# --gender      Specify gender for targeted recommendations (default: auto-detect)
```

## How It Works

1. **Face Detection and Preprocessing**:
   - Detects the face in the input image
   - Aligns the face based on eye positions
   - Removes the background to isolate facial features

2. **Feature Extraction**:
   - Identifies 68 facial landmarks
   - Calculates key proportions:
     - Face width to height ratio
     - Jaw width to face width ratio
     - Cheekbone to jaw width ratio
     - Hairline shape and forehead size

3. **Face Shape Classification**:
   - Classifies the face into one of five common shapes:
     - Oval
     - Round
     - Square
     - Heart
     - Rectangle/Oblong

4. **Hairstyle Recommendation**:
   - Matches the identified face shape with suitable hairstyle attributes
   - Considers other factors (hair type, density, etc. if provided)
   - Generates personalized recommendations with visual examples

## Technical Details

### Face Landmark Detection
The system uses the face_recognition library to detect 68 facial landmarks that serve as the foundation for all measurements and analysis.

### Background Removal
The MODNet (Matting Objective Decomposition Network) implementation is used to isolate the portrait from background elements, ensuring accurate facial measurements.

### Face Shape Classification
A VGG16-based neural network trained on thousands of labeled face images classifies faces into the appropriate shape category with high accuracy.

### Image Processing Pipeline
1. **Input**: Portrait image capture
2. **Preprocessing**: Face alignment and standardization
3. **Background Removal**: Isolation of facial features
4. **Landmark Detection**: Identification of key facial points
5. **Measurement**: Calculation of facial proportions
6. **Classification**: Determination of face shape
7. **Recommendation**: Selection of suitable hairstyles

## Model Training

The face shape classifier was trained on a dataset of diverse facial images with annotated face shapes. Transfer learning was applied using the VGG16 architecture pretrained on ImageNet, with custom classification layers added and fine-tuned for face shape detection.

## Future Improvements

- Additional demographic considerations for more targeted recommendations
- Hairstyle virtual try-on functionality
- Hair texture and color analysis for improved recommendations
- Mobile application development
- Integration with salon booking systems

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Face landmark detection utilizes the face_recognition library
- Background removal powered by MODNet
- VGG16 model adapted from the original VGGFace implementation
