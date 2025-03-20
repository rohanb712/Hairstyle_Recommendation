
import cv2
import dlib
import numpy as np
import math
from typing import List, Tuple, Optional, Union, Any
           
def octagonal_boundary_extraction(height: int, width: int) -> np.ndarray:
    """
    Returns 8 points on the boundary of a rectangle.
    
    Args:
        height: Height of the rectangle
        width: Width of the rectangle
        
    Returns:
        Array of 8 boundary points
    """
    boundary_vertices = []
    boundary_vertices.append((0, 0))
    boundary_vertices.append((width/2, 0))
    boundary_vertices.append((width-1, 0))
    boundary_vertices.append((width-1, height/2))
    boundary_vertices.append((width-1, height-1))
    boundary_vertices.append((width/2, height-1))
    boundary_vertices.append((0, height-1))
    boundary_vertices.append((0, height/2))
    return np.array(boundary_vertices, dtype=np.float)


def point_within_boundary_enforcer(point: Tuple[float, float], width: int, height: int) -> Tuple[float, float]:
    """
    Constrains points to be inside boundary.
    
    Args:
        point: The point to constrain
        width: Width of the boundary
        height: Height of the boundary
        
    Returns:
        Constrained point
    """
    constrained_point = (min(max(point[0], 0), width - 1), min(max(point[1], 0), height - 1))
    return constrained_point

def landmark_extractor_from_dlib(shape: dlib.full_object_detection) -> List[Tuple[int, int]]:
    """
    Convert Dlib shape detector object to list of tuples.
    
    Args:
        shape: Dlib shape detector object
        
    Returns:
        List of points (x, y) tuples
    """
    facial_keypoints = []
    for landmark_point in shape.parts():
        coordinate_pair = (landmark_point.x, landmark_point.y)
        facial_keypoints.append(coordinate_pair)
    return facial_keypoints

def equilateral_transform_calculator(input_points: np.ndarray, output_points: np.ndarray) -> np.ndarray:
    """
    Compute similarity transform given two sets of two points.
    OpenCV requires 3 pairs of corresponding points.
    We are faking the third one.
    
    Args:
        input_points: Array of input points
        output_points: Array of output points
        
    Returns:
        Transformation matrix
    """
    # Constants for 60-degree triangle calculation
    sine_60deg = math.sin(60*math.pi/180)
    cosine_60deg = math.cos(60*math.pi/180)

    source_points = np.copy(input_points).tolist()
    target_points = np.copy(output_points).tolist()

    # Calculate third point for source equilateral triangle
    x_source = cosine_60deg*(source_points[0][0] - source_points[1][0]) - sine_60deg*(source_points[0][1] - source_points[1][1]) + source_points[1][0]
    y_source = sine_60deg*(source_points[0][0] - source_points[1][0]) + cosine_60deg*(source_points[0][1] - source_points[1][1]) + source_points[1][1]

    source_points.append([np.int_(x_source), np.int_(y_source)])

    # Calculate third point for target equilateral triangle
    x_target = cosine_60deg*(target_points[0][0] - target_points[1][0]) - sine_60deg*(target_points[0][1] - target_points[1][1]) + target_points[1][0]
    y_target = sine_60deg*(target_points[0][0] - target_points[1][0]) + cosine_60deg*(target_points[0][1] - target_points[1][1]) + target_points[1][1]

    target_points.append([np.int_(x_target), np.int_(y_target)])

    # Calculate the rigid transformation
    transformation_matrix = cv2.estimateRigidTransform(np.array([source_points]), np.array([target_points]), False)
    return transformation_matrix

def face_standardizer(output_dimensions: Tuple[int, int], input_image: np.ndarray, input_landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizes a facial image to a standard size given by output_dimensions.
    Normalization is done based on Dlib's landmark points passed as input_landmarks.
    After normalization, left corner of the left eye is at (0.3 * w, h/3)
    and right corner of the right eye is at (0.7 * w, h/3) where w and h
    are the width and height of output_dimensions.
    
    Args:
        output_dimensions: Desired output size (width, height)
        input_image: Input image
        input_landmarks: Facial landmarks
        
    Returns:
        Tuple of (normalized image, normalized landmarks)
    """
    height, width = output_dimensions

    # Eye corners from landmark points (36 = left eye left corner, 45 = right eye right corner)
    eye_corners_source = [input_landmarks[36], input_landmarks[45]]

    # Target positions for eye corners after normalization
    eye_corners_destination = [(np.int_(0.3 * width), np.int_(height/3)), 
                              (np.int_(0.7 * width), np.int_(height/3))]

    # Calculate similarity transform
    transform_matrix = equilateral_transform_calculator(eye_corners_source, eye_corners_destination)
    output_image = np.zeros(input_image.shape, dtype=input_image.dtype)

    # Apply similarity transform to input image
    output_image = cv2.warpAffine(input_image, transform_matrix, (width, height))

    # Reshape input_landmarks from numLandmarks x 2 to numLandmarks x 1 x 2
    landmarks_reshaped = np.reshape(input_landmarks, (input_landmarks.shape[0], 1, input_landmarks.shape[1]))
    
    # Apply similarity transform to landmarks
    output_landmarks = cv2.transform(landmarks_reshaped, transform_matrix)

    # Reshape output_landmarks back to numLandmarks x 2
    output_landmarks = np.reshape(output_landmarks, (input_landmarks.shape[0], input_landmarks.shape[1]))

    return output_image, output_landmarks

def nearest_point_finder(points_array: np.ndarray, query_point: np.ndarray) -> int:
    """
    Find the point closest to an array of points.
    points_array is a Nx2 and point is 1x2 ndarray.
    
    Args:
        points_array: Array of points
        query_point: Query point
        
    Returns:
        Index of the closest point
    """
    distance_vector = np.linalg.norm(points_array-query_point, axis=1)
    closest_point_index = np.argmin(distance_vector)
    return closest_point_index


def is_point_inside_rectangle(rectangle_coords: Tuple[int, int, int, int], test_point: Tuple[int, int]) -> bool:
    """
    Check if a point is inside a rectangle.
    
    Args:
        rectangle_coords: Rectangle coordinates (left, top, right, bottom)
        test_point: Point to test (x, y)
        
    Returns:
        True if point is inside rectangle, False otherwise
    """
    if test_point[0] < rectangle_coords[0]:
        return False
    elif test_point[1] < rectangle_coords[1]:
        return False
    elif test_point[0] > rectangle_coords[2]:
        return False
    elif test_point[1] > rectangle_coords[3]:
        return False
    return True


def triangulation_mesh_generator(bounding_rectangle: Tuple[int, int, int, int], 
                               vertex_points: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    """
    Calculate Delaunay triangles for set of points.
    Returns the vector of indices of 3 points for each triangle.
    
    Args:
        bounding_rectangle: Rectangle coordinates (left, top, right, bottom)
        vertex_points: List of points
        
    Returns:
        List of triangles as triplets of point indices
    """
    # Create subdivision for Delaunay triangulation
    subdivision = cv2.Subdiv2D(bounding_rectangle)

    # Insert all points into subdivision
    for vertex in vertex_points:
        subdivision.insert((vertex[0], vertex[1]))

    # Get raw triangulation data
    triangle_list = subdivision.getTriangleList()

    # Process triangles to get indices
    delaunay_triangles = []

    for triangle in triangle_list:
        # Extract the three points of this triangle
        triangle_points = []
        triangle_points.append((triangle[0], triangle[1]))
        triangle_points.append((triangle[2], triangle[3]))
        triangle_points.append((triangle[4], triangle[5]))

        # Check if all triangle vertices are within the bounding rectangle
        if (is_point_inside_rectangle(bounding_rectangle, triangle_points[0]) and
            is_point_inside_rectangle(bounding_rectangle, triangle_points[1]) and
            is_point_inside_rectangle(bounding_rectangle, triangle_points[2])):
            
            # Find indices of the triangle vertices in the original point list
            vertex_indices = []
            
            for vertex_idx in range(0, 3):
                for point_idx in range(0, len(vertex_points)):
                    if (abs(triangle_points[vertex_idx][0] - vertex_points[point_idx][0]) < 1.0 and 
                        abs(triangle_points[vertex_idx][1] - vertex_points[point_idx][1]) < 1.0):
                        vertex_indices.append(point_idx)
                        
            # Store triangulation as list of vertex indices if all 3 were found
            if len(vertex_indices) == 3:
                delaunay_triangles.append((vertex_indices[0], vertex_indices[1], vertex_indices[2]))

    return delaunay_triangles

def geometric_transformation_applier(source_image: np.ndarray, 
                                  source_triangle: List[Tuple[int, int]], 
                                  target_triangle: List[Tuple[int, int]], 
                                  output_size: Tuple[int, int]) -> np.ndarray:
    """
    Apply affine transform calculated using source_triangle and target_triangle to source_image,
    and output an image of size output_size.
    
    Args:
        source_image: Source image
        source_triangle: Source triangle points
        target_triangle: Target triangle points
        output_size: Size of output image
        
    Returns:
        Transformed image
    """
    # Calculate affine transform
    warp_matrix = cv2.getAffineTransform(np.float32(source_triangle), np.float32(target_triangle))

    # Apply the affine transform
    destination_image = cv2.warpAffine(
        source_image, 
        warp_matrix, 
        (output_size[0], output_size[1]), 
        None,
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_REFLECT_101
    )

    return destination_image

def triangle_morphing_blender(image1: np.ndarray, image2: np.ndarray, triangle1: List[Tuple[int, int]], triangle2: List[Tuple[int, int]]):
    """
    Warps and alpha blends triangular regions from img1 and img2 to img.
    
    Args:
        image1: First input image
        image2: Second input image (target)
        triangle1: Triangle in the first image
        triangle2: Triangle in the second image
    """
    # Find bounding rectangles for both triangles
    rect1 = cv2.boundingRect(np.float32([triangle1]))
    rect2 = cv2.boundingRect(np.float32([triangle2]))

    # Offset points by left top corner of the respective rectangles
    triangle1_rect_space = []
    triangle2_rect_space = []
    triangle2_rect_int = []

    for i in range(0, 3):
        triangle1_rect_space.append(((triangle1[i][0] - rect1[0]), (triangle1[i][1] - rect1[1])))
        triangle2_rect_space.append(((triangle2[i][0] - rect2[0]), (triangle2[i][1] - rect2[1])))
        triangle2_rect_int.append(((triangle2[i][0] - rect2[0]), (triangle2[i][1] - rect2[1])))

    # Create mask for triangle in second image
    alpha_mask = np.zeros((rect2[3], rect2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(alpha_mask, np.int32(triangle2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Extract small rectangular patches from the source image
    image1_rect = image1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]

    rect_size = (rect2[2], rect2[3])

    # Warp source rectangle to destination rectangle
    image2_rect = geometric_transformation_applier(image1_rect, triangle1_rect_space, triangle2_rect_space, rect_size)

    # Apply mask to warped image
    image2_rect = image2_rect * alpha_mask

    # Copy triangle region back into second image
    image2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = (
        image2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] * 
        ((1.0, 1.0, 1.0) - alpha_mask)
    )
    image2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] += image2_rect

def facial_feature_detector(face_detector: dlib.fhog_object_detector, 
                          landmark_detector: dlib.shape_predictor, 
                          image: np.ndarray, 
                          FACE_DOWNSAMPLE_RATIO: float = 1) -> List[Tuple[int, int]]:
    """
    Detect facial landmarks in an image.
    
    Args:
        face_detector: Dlib face detector
        landmark_detector: Dlib landmark detector
        image: Input image
        FACE_DOWNSAMPLE_RATIO: Ratio to downsample face for detection
        
    Returns:
        List of detected facial landmark points
    """
    detected_points = []
    
    # Scale down image for face detection to speed up processing
    image_small = cv2.resize(
        image, 
        None,
        fx=1.0/FACE_DOWNSAMPLE_RATIO, 
        fy=1.0/FACE_DOWNSAMPLE_RATIO, 
        interpolation=cv2.INTER_LINEAR
    )
    
    # Detect faces in the small image
    face_detections = face_detector(image_small, 0)
    
    if len(face_detections) > 0:
        max_area = 0
        max_rectangle = None
        
        # Find the largest face (highest area)
        for face in face_detections:
            if face.area() > max_area:
                max_area = face.area()
                max_rectangle = [face.left(),
                               face.top(),
                               face.right(),
                               face.bottom()
                              ]
        
        # Create a dlib rectangle & adjust for original image size
        rect = dlib.rectangle(*max_rectangle)
        scaled_rect = dlib.rectangle(
            int(rect.left()*FACE_DOWNSAMPLE_RATIO),
            int(rect.top()*FACE_DOWNSAMPLE_RATIO),
            int(rect.right()*FACE_DOWNSAMPLE_RATIO),
            int(rect.bottom()*FACE_DOWNSAMPLE_RATIO)
        )
        
        # Detect landmarks on the original sized image
        landmarks = landmark_detector(image, scaled_rect)
        
        # Convert to list of points
        detected_points = landmark_extractor_from_dlib(landmarks)
        
    return detected_points

def piecewise_affine_warper(input_image: np.ndarray, 
                          source_points: np.ndarray, 
                          target_points: np.ndarray, 
                          triangulation: List[Tuple[int, int, int]]) -> np.ndarray:
    """
    Warps an image in a piecewise affine manner.
    The warp is defined by the movement of landmark points specified by source_points
    to a new location specified by target_points. The triangulation between points is specified
    by their indices in triangulation.
    
    Args:
        input_image: Input image
        source_points: Source points
        target_points: Target points
        triangulation: Triangulation between points
        
    Returns:
        Warped image
    """
    height, width, channels = input_image.shape
    
    # Create output image (initially all zeros)
    output_image = np.zeros(input_image.shape, dtype=input_image.dtype)

    # Process each triangle
    for triangle_index in range(0, len(triangulation)):
        # Get triangle vertices for source and target
        triangle_in = []
        triangle_out = []

        for vertex_index in range(0, 3):
            # Get the vertex point from triangle indices
            point_in = source_points[triangulation[triangle_index][vertex_index]]
            # Ensure the vertex is inside the image boundary
            point_in = point_within_boundary_enforcer(point_in, width, height)
            # Add to source triangle vertices
            triangle_in.append(point_in)

            # Do the same for target/output triangles
            point_out = target_points[triangulation[triangle_index][vertex_index]]
            point_out = point_within_boundary_enforcer(point_out, width, height)
            triangle_out.append(point_out)

        # Warp pixels from source to destination triangle
        triangle_morphing_blender(input_image, output_image, triangle_in, triangle_out)
    
    return output_image