import os
import numpy as np
import cv2
import time
import sys

'''
This file contains the code to load the data from the datasets.
Each dataset is loaded using a different function.
'''

current_path = os.path.dirname(os.path.abspath(__file__)) # Get the current path
data_path = current_path  # Get the data directory


def load_kitti() -> (np.ndarray, np.ndarray, int):
    """
    Loads the KITTI dataset.

    #TODO: Select which data to return and update the docstring.
    Returns:
        Dict: A dictionary containing the keys:
        - "Images" -> Images (Nx640x480) [numpy.ndarray]
        - "K" -> K matrix (3x3) [numpy.ndarray]
        - "Homogeneous_Pose_Mat" -> Ground truth Homogeneous Pose Matrices (Nx4x4) [numpy.ndarray]
        - "Num_Images" -> Number of images [int]

    """

    kitti_path = os.path.join(data_path, 'kitti')
    kitti_image_path = os.path.join(kitti_path, '05', "image_0")
    assert 'kitti_path' in locals(), "kitti_path variable is not defined"


    # Load the ground truth pose values (3x4 matrix)
    ground_truth = np.loadtxt(os.path.join(kitti_path, 'poses', '05.txt'))
    ground_truth = ground_truth.reshape(-1, 3, 4)

    # Convert the ground truth pose values to homogeneous coordinates (4x4 matrix)
    ground_truth_homogeneous = np.zeros((ground_truth.shape[0], 4, 4))
    ground_truth_homogeneous[:, :3, :] = ground_truth
    ground_truth_homogeneous[:, 3, 3] = 1

    # Load Image_0
    images = [cv2.imread(os.path.join(kitti_image_path, filename), cv2.IMREAD_GRAYSCALE) for filename in os.listdir(kitti_image_path)]
    images = np.array(images)


    num_of_images = len(images)

    K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                    [0, 7.188560000000e+02, 1.852157000000e+02],
                    [0, 0, 1]])
    

    # Build the dictionary
    out = {
        "Images": images,
        "K": K,
        "Homogeneous_Pose_Mat": ground_truth_homogeneous,
        "Num_Images": num_of_images
    }

    return out

def load_malaga():
    """
    Loads the Parking dataset.
    
    Returns:
        Dict: A dictionary containing the keys:
        - "Images" -> Images (Nx640x480) [numpy.ndarray]
        - "K" -> K matrix (3x3) [numpy.ndarray]
        - "Num_Images" -> Number of images [int]
        - "GPS" -> GPS data (Nx3) [numpy.ndarray]

    """


    malaga_path = os.path.join(data_path, 'malaga-urban-dataset-extract-07')
    malaga_image_path = os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images')
    assert 'malaga_path' in locals(), "malaga_path variable is not defined"

    # Load the images
    images = [cv2.imread(os.path.join(malaga_image_path, filename), cv2.IMREAD_GRAYSCALE) for filename in os.listdir(malaga_image_path) if filename.endswith('left.jpg')]
    images = np.array(images)
    
    num_of_images = len(images)

    # K matrix
    K = np.array([[621.18428, 0, 404.0076],
                  [0, 621.18428, 309.05989],
                  [0, 0, 1]])
    
    # Load GPS data (Get Timestamps and Local Position)
    gps_path = os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_all-sensors_GPS.txt')
    gps_data = np.loadtxt(gps_path, skiprows=1, usecols=(0, 8, 9))

    # Load IMU data (Get Timestamps, XYZ Acceleration and Angular Velocity)
    imu_path = os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_all-sensors_IMU.txt')
    imu_data = np.loadtxt(imu_path, skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6))
    
    # Build the dictionary
    out = {
        "Images": images,
        "K": K,
        "Num_Images": num_of_images,
        "GPS": gps_data,
        "IMU": imu_data
    }

    return out



def load_parking():
    """
    Loads the Parking dataset.

    Returns:
        Dict: A dictionary containing the keys:
        - "Images" -> Images (Nx640x480) [numpy.ndarray]
        - "K" -> K matrix (3x3) [numpy.ndarray]
        - "Homogeneous_Pose_Mat" -> Ground truth Homogeneous Pose Matrices (Nx4x4) [numpy.ndarray]
        - "Num_Images" -> Number of images [int]

    """

    parking_path = os.path.join(data_path, 'parking')
    parking_image_path = os.path.join(parking_path, 'images')    
    assert 'parking_path' in locals(), "parking_path variable is not defined"

    # Load the images
    images = [cv2.imread(os.path.join(parking_image_path, filename), cv2.IMREAD_GRAYSCALE) for filename in os.listdir(parking_image_path) if filename.endswith('.png')]
    images = np.array(images)

    num_of_images = len(images) - 1

    # Side note: the following code is a bit convoluted, but it didn't work with just np.loadtxt
    K_rows = []  # List to accumulate rows of K
    with open(os.path.join(parking_path, 'K.txt'), 'r') as file:
        lines = file.readlines()

        for line in lines:
            try:
                values = list(map(float, line.strip().rstrip(',').split(',')))
                K_rows.append(values)
            except ValueError as e:
                print(f"Error converting line '{line.strip()}' to float: {e}")

    K = np.array(K_rows)


    # Load the ground truth pose values (3x4 matrix)
    ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
    ground_truth = ground_truth.reshape(-1, 3, 4)

    # Convert the ground truth pose values to homogeneous coordinates (4x4 matrix)
    ground_truth_homogeneous = np.zeros((ground_truth.shape[0], 4, 4))
    ground_truth_homogeneous[:, :3, :] = ground_truth
    ground_truth_homogeneous[:, 3, 3] = 1

    # Build the dictionary
    out = {
        "Images": images,
        "K": K,
        "Homogeneous_Pose_Mat": ground_truth_homogeneous,
        "Num_Images": num_of_images
    }

    return out


def loading_animation_continuous(text, event):
    """
    Create a simple terminal animation with points that appear and disappear.
    Keeps running until the event is set.

    Parameters:
    text (str): The text to display before the animation
    event (threading.Event): Event that, when set, stops the animation
    """
    while not event.is_set():
        for num_dots in range(4):
            sys.stdout.write('\r' + text + ' ' + '.' * num_dots + ' ' * (3 - num_dots))
            sys.stdout.flush()
            time.sleep(0.5)
        if event.is_set():
            break
