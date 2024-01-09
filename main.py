# Vision Algorithms for Mobile Robotics final project

import os

import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import threading
from IPython.display import display, clear_output

# Internal Imports
from data.data_loader import load_kitti, load_malaga, load_parking, loading_animation_continuous
from visual_odometry import Visual_Odometry

np.set_printoptions(precision=2, suppress=True)



# Print of initial statements
print('\n --- Vision Algorithms for Mobile Robotics Mini Project --- ')
print('Authors: Axel Barbelanne, Andre Gomes, Jose Lavariega, Simone Manni\n')

## Instructions:
#   - Place the unmodified datasets folders in the 'data' folder
#   - Change parameter ds to switch between datatsets
#   - Change parameters n_imgs to process a smaller number of images
#   - Press 'c' after the plot opens to continue
#   - Place 'q' to close the plot during continuous operation


# Setup
# Change ds to use a different dataset
# 0: KITTI, 1: Malaga, 2: parking
ds = 2

# To make the following code work place the unmodified datasets folder in the 'data' folder
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)  # Get the parent directory
data_path = os.path.join(current_path, 'data')  # Get the data directory


### Data loading ###
print('1) Data loading\n')
stop_loading_event = threading.Event()

if ds == 0:
    # Start of animation on separate thread
    loading_thread = threading.Thread(target=loading_animation_continuous, args=('Loading KITTI', stop_loading_event))
    loading_thread.start()

    data = load_kitti()
    images = data["Images"]
    K = data["K"]
    homogeneous_pose_mat = data["Homogeneous_Pose_Mat"]
    num_images = data["Num_Images"]

    stop_loading_event.set()
    loading_thread.join()

    print("\nLoading completed")

elif ds == 1:

    # Start of animation on separate thread
    loading_thread = threading.Thread(target=loading_animation_continuous, args=('Loading Malaga', stop_loading_event))
    loading_thread.start()

    data = load_malaga()
    images = data["Images"]
    K = data["K"]
    num_images = data["Num_Images"]

    stop_loading_event.set()
    loading_thread.join()

    print("\nLoading completed")

elif ds == 2:
    # Start of animation on separate thread
    loading_thread = threading.Thread(target=loading_animation_continuous, args=('Loading Parking', stop_loading_event))
    loading_thread.start()

    data = load_parking()

    images = data["Images"]
    K = data["K"]
    homogeneous_pose_mat = data["Homogeneous_Pose_Mat"]
    num_images = data["Num_Images"]

    stop_loading_event.set()
    loading_thread.join()

    print("\nLoading completed")

else:
    assert False

### Initialization ###
print('\n2) Initialization\n')
feature_detector = cv2.SIFT_create()
feature_matcher = cv2.BFMatcher(cv2.NORM_L2, True)

params = {"Feature_Detector": feature_detector,
            "Feature_Matcher": feature_matcher,
            "K": K,
            "dataset": ds}

vo = Visual_Odometry(params)
if ds == 0:
    vo.bootstrap(images[0], images[2])
else:
    vo.automatic_bootstrap(images[0:20])

print('Initialization completed')


### Continuous operation ###
print('\n3) Continous operation')

plt.ion()
fig, plots = vo.setup_plot_all(images[0])
manager = plt.get_current_fig_manager()

c_pressed = False  # Initialize a flag to track if 'c' is pressed

def on_key_c(event):
    global c_pressed
    if event.key == 'c':
        c_pressed = True  # Set the flag to True when 'c' is pressed

# Connect the key press event
fig.canvas.mpl_connect('key_press_event', on_key_c)
plt.show()

# Wait for the 'c' key to be pressed before continuing
while not c_pressed:
    plt.pause(0.1)  # Use a short pause to keep checking for the key press

print('Proceeding to updating plot . . .')

# Change instruction on bottom of figure
instructions_text = plots[-1]
instructions_text.set_text('Press ''q'' to quit')

# Change this parameter to display a different number of images
n_imgs = len(images)
# n_imgs = 100

# Plot 0: Image keypoints with keypoints
# No additional variable needed

# Plot 1: Keypoints statistics
num_kp = np.array([], dtype=int)
num_candidate_kp = np.array([], dtype=int)
num_added_kp = np.array([], dtype=int)

# Plot 2: Global trajectory
trajectory = np.empty((0, 3))

# Plot 3: Local trajectory
local_trajectory = None
R = None
new_points_3d = None
old_points_3d = np.empty((0, 3))
num_3d_points = np.array([], dtype=int)

quit_pressed = False
def on_key_q(event):
    global quit_pressed  # Reference the global flag
    if event.key == 'q':
        quit_pressed = True  # Set the flag when 'q' is pressed

fig.canvas.mpl_connect('key_press_event', on_key_q)

for idx, image in enumerate(images[:n_imgs]):
    if idx == 0:
        # Plot 0
        data0 = (None, vo.S, image)

        # Plot 1
        num_kp = np.append(num_kp, vo.S.shape[0])
        num_candidate_kp = np.append(num_candidate_kp, 0)
        num_added_kp = np.append(num_added_kp, vo.added_kp)
        data1 = (num_kp, num_candidate_kp, num_added_kp)

        # Plot 2
        trajectory = np.append(trajectory, np.zeros((1, 3)), axis=0)
        data2 = trajectory

        # Plot 3
        data3 = (trajectory, np.eye(3), vo.P, None)
        old_points_3d = np.append(old_points_3d, vo.P, axis=0)
        num_3d_points = np.append(num_3d_points, len(vo.P))

        data = (data0, data1, data2, data3)
        vo.update_plot(fig, plots, data, 0)
        continue

    kp_prev = vo.S
    R, t = vo.process_new_frame(image)
    kp_curr = vo.S
    data0 = (kp_prev, kp_curr, image)

    num_kp = np.append(num_kp, vo.S.shape[0])
    num_candidate_kp = np.append(num_candidate_kp, vo.C.shape[0])
    num_added_kp = np.append(num_added_kp, vo.added_kp)
    data1 = (num_kp, num_candidate_kp, num_added_kp)

    trajectory = np.append(trajectory, t.reshape(1, -1), axis=0)
    data2 = trajectory

    # Take last n positions if available
    local_trajectory = trajectory
    if idx > vo.n_last_frames:
        local_trajectory = local_trajectory[-vo.n_last_frames:]
        old_points_3d = old_points_3d[num_3d_points[0]:]
        num_3d_points = num_3d_points[1:]


    data3 = (local_trajectory, R, vo.P, old_points_3d)
    old_points_3d = np.append(old_points_3d, vo.P, axis=0)
    num_3d_points = np.append(num_3d_points, len(vo.P))

    data = (data0, data1, data2, data3)
    vo.update_plot(fig, plots, data, idx)

    if quit_pressed:
        break

plt.pause(4)

# Plot of comparison between estimated trajectory and ground truth
if ds == 0 or ds == 2:
    vo.plot_final_trajectory_ground_truth(trajectory, homogeneous_pose_mat[:len(trajectory)])
    manager = plt.get_current_fig_manager()
elif ds == 1:
    vo.plot_final_trajectory_without_ground_truth(trajectory)
    manager = plt.get_current_fig_manager()

plt.show(block=True)