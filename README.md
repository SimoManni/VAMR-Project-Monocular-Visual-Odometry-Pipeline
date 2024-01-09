# Monocular Visual Odometry Project

## Course Information

**Institution:** University of Zurich (UZH) and Swiss Federal Institute of Technology (ETH)  
**Course:** Vision Algorithms for Mobile Robotics  
**Instructor:** Prof. Davide Scaramuzza

## Project Overview

This project is a part of the Vision Algorithms for Mobile Robotics course, focusing on implementing and understanding the fundamentals of monocular visual odometry. The goal is to develop an algorithm capable of estimating the 3D motion of a single camera moving through a static environment. This technique is critical in the domains of robotics and autonomous vehicles, where understanding the movement relative to the environment is crucial.

## Objectives

* **Implement feature detection, matching, and tracking.** The algorithm utilizes feature detection algorithms, such as SIFT or SURF, to identify and match keypoints across consecutive images. This process allows for the establishment of correspondences between points in different frames, which is essential for estimating camera motion.

* **Estimate camera motion from a sequence of images.** Based on the matched features and their relative positions in the images, the algorithm estimates the 3D motion of the camera. This involves calculating the camera's translation and rotation between frames, enabling the reconstruction of the camera's path through the environment.

* **Understand the principles of monocular visual odometry and its applications.** The project delves into the theoretical foundations of monocular visual odometry, exploring the underlying principles and challenges involved in estimating camera motion from a single perspective. This knowledge provides a deeper understanding of the algorithm's capabilities and limitations.

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

Device specifications on which the code was run: 
- Processor: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
- RAM: 16,0 GB (15,8 GB usable)
- Operating system: Windows 11
- Maximum number of threads used: 2

