# Monocular Visual Odometry Project

## Course Information

**Institution:** University of Zurich (UZH) and Swiss Federal Institute of Technology (ETH)  
**Course:** Vision Algorithms for Mobile Robotics  
**Instructor:** Prof. Davide Scaramuzza

## Project Overview

This project is a part of the Vision Algorithms for Mobile Robotics course, focusing on implementing and understanding the fundamentals of monocular visual odometry. The goal is to develop an algorithm capable of estimating the 3D motion of a single camera moving through a static environment. This technique is critical in the domains of robotics and autonomous vehicles, where understanding the movement relative to the environment is crucial.

## Objectives

The primary objective of this project is to implement and evaluate a simple monocular visual odometry (VO) pipeline with key features, including:

* **Initialization of 3D Landmarks:** This involves the extraction of initial sets of 2D ↔ 3D correspondences from the first frames of a sequence and bootstrapping the initial camera poses and landmarks.

* **Keypoint Tracking Between Frames:** The project requires tracking keypoints between two consecutive frames, which is crucial for maintaining the continuity and accuracy of the visual odometry pipeline.

* **Pose Estimation Using Established 2D ↔ 3D Correspondences:** Accurate pose estimation is essential for understanding the movement and orientation of the camera in space over time.

* **Triangulation of New Landmarks:** As the camera moves through the environment, it is necessary to continuously triangulate and add new landmarks to the model to ensure comprehensive environmental mapping.

* **Use of Provided Datasets for Testing:** The project will utilize three specific datasets (parking, KITTI1, and Malaga2) for testing and validating the VO pipeline, each offering different challenges and scenarios.


## Installation

### Prerequisites

* **Python 3.x:** The project is developed using Python 3.x. Ensure Python 3.x is installed on your system.

* **OpenCV:** OpenCV is a powerful library for computer vision. Install OpenCV using your system's package manager or by downloading the official OpenCV installer.

* **Numpy:** Numpy is a fundamental library for scientific computing in Python. Install Numpy using your system's package manager or by downloading the official Numpy installer.

### Setup

1. **Clone the repository:** Clone the project's repository using the following command:

```sh
git clone https://github.com/JoseLavariega/VisualAlgosProject.git
```
2. Install the required packages:
```sh
pip install -r requirements.txt
```


## How to Run

1. **Add data:** Place the unmodified image sequences folders inside the 'data' folder

2. **Open the main.py file:** Open the `main.py` file in a text editor.

3. **Execute the main script:** Run the  `main.py` and wait for a figure to show up. The waiting time might vary depending on the performance of the computer.

4.  **Switch dataset:** Locate the 'ds' variable at line 36 of the `main.py` file and change the value between 0, 1, 2 to switch between datasets. 

5.  **Change number of processed images:** Locate the 'n_imgs' variable at line 149 of the `main.py` file and change the value to process less images. 

6. **Figure instructions:** When the first figure is shown, press 'c' to initialize the continuous plotting. Press 'q' to break and display the estimated trajectory.


The script will process the specified image sequence, displaying in subplots important information being processed in real time. After that, the script will display the estimated camera trajectory together with the ground truth, if available, to assess the accuracy of the pipeline. 

## Results
Here are presented the results on the KITTI dataset, a popular dataset to benchmark the performance of the pipeline.
First, the continuous operation of the pipeline is displayed by visualasing the processed images, together with the identified keypoints, a plot of the number of processed points and a top down view of the global trajecotory as well as the local trajectory together with the 3D landmarks. 

<img src="https://github.com/SimoManni/VAMR-Project-Monocular-Visual-Odometry-Pipeline/assets/151052936/013aa589-ba69-4c6c-a419-66c9d8ee4eb5" alt="Screenshot 2024-02-07 210533" style="max-width: 50%; display: block; margin: 0 auto;">

<img src="https://github.com/SimoManni/VAMR-Project-Monocular-Visual-Odometry-Pipeline/assets/151052936/ad6e6078-134f-4740-9ec7-d815c6aa48cc" alt="KittiGIF" style="max-width: 100%; display: block; margin: 0 auto;">


Our visual odometry (VO) pipeline was assessed using the KITTI dataset, comparing the estimated trajectory against the ground truth. As observed in the provided image, there is a noticeable scale ambiguity problem, as well as some scale drift over time which leads to discrepancies between the estimated trajectory and the ground truth.

### Scale Ambiguity
The scale ambiguity issue arises from the monocular VO setup, where the absence of depth information from a single camera leads to an inability to recover the absolute scale of the scene. This has resulted in a trajectory that, while directionally similar to the ground truth, differs in the actual distance traveled.

### Scale Drift
Over time, the VO pipeline accumulates errors, evident from the divergence of the estimated trajectory from the ground truth. This drift is a cumulative effect of several factors, including error in feature tracking, camera motion estimation, and map updates.
