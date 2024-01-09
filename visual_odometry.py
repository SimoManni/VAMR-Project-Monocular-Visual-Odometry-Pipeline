# Imports
from typing import Any, Union, Tuple, List
import numpy as np
import cv2
from scipy.optimize import least_squares
import matplotlib
import matplotlib.pyplot as plt
import os

class Visual_Odometry:

    ## Initialization
    def __init__(self, params) -> None:
        """
        Initializes the Vision Odometry object.
        
        Args:
            feature_detector (cv2.Feature2D): A feature detector object.
            matcher (cv2.DescriptorMatcher): A descriptor matcher object.
            K (numpy.ndarray): The camera matrix.
        """
        self.feature_detector = params["Feature_Detector"]
        self.matcher = params["Feature_Matcher"]
        self.K = params["K"]
        self.dataset = params["dataset"]

        self.descriptor_size = 128  # TODO: This is hardcoded for SIFT. Make it work for any descriptor.

        ## State initialization of the Vision Odometry
        # Keypoints array
        self.S = np.zeros((0, 2), dtype=np.float32)
        # 3D points array
        self.P = np.zeros((0, 3), dtype=np.float32)
        # Candidate keypoints array
        self.C = np.zeros((0, 2), dtype=np.float32)
        # First observation of candidate keypoints array
        self.F = np.zeros((0, 2), dtype=np.float32)
        # Camera poses of first observation of candidate keypoints array
        self.T = np.zeros((0, 4, 4), dtype=np.float32)

        # For plotting
        # Added keypoints during last frame
        self.added_kp: int = 0
        # Number of last frames to consider for plotting
        self.n_last_frames = 20

        # Last image
        self.prev_image = None

        # Parameters
        # Lucas-Kanade parameters
        self.klt_params = {
            'winSize': (10, 10),
            'maxLevel': 5,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.05)
        }

        # PnP RAMSAC parameters
        if self.dataset == 0: #Kitti
            iter = 80
        elif self.dataset == 1:
            iter = 150
        elif self.dataset == 2:
            iter = 80

        self.pnp_ransac_params = {
            "iterationsCount": iter, # High number to prevent not finding inliers
            "reprojectionError": 1.0,
            "confidence": 0.99,
            "distCoeffs": None,
            "flags": cv2.SOLVEPNP_ITERATIVE,
            "useExtrinsicGuess": False
        }

        if self.dataset == 0:  # Kitti
            angle = 30/180 * np.pi
        elif self.dataset == 1:
            angle = 27/180 * np.pi
        elif self.dataset == 2:
            angle = 30 / 180 * np.pi

        self.base_line_angle = angle
        self.scale = 0


    ## Bootstrapping
    def automatic_bootstrap(self, images):
        """
        Bootstraps the Visual Odometry class automatically by applying an heuristic on the triangulated 3D points and
        generates the initial features and initial point cloud.

        Args:
            images (numpy.ndarray): A 3-dimensional array of the initial images

        """

        image_1 = images[0]
        self.prev_image = image_1
        kp_1, des_1 = self.get_image_keypoints_and_descriptors(image_1)

        for image_2 in images[1:]:
            kp_2, des_2 = self.get_image_keypoints_and_descriptors(image_2)
            # Indices of the inlier points
            kp_1_id = np.arange(len(kp_1))
            kp_2_id = np.arange(len(kp_2))

            T, points_3d, matches_1_id, matches_2_id = self.get_relative_pose_and_3d_points_from_features(kp_1, des_1, kp_2, des_2)

            # Rule of thumb for initilization
            if np.linalg.norm(T[:3, 3]) / np.mean(points_3d[:,2]) > 0.10:
                # Update the inlier points indices
                kp_1_id = kp_1_id[matches_1_id]
                kp_2_id = kp_2_id[matches_2_id]

                # Update state of the vision
                self.S = kp_1[kp_1_id]
                self.P = points_3d
                self.scale = np.mean(np.linalg.norm(points_3d, axis=1))

                break


        assert self.S.shape[0] == self.P.shape[0], "The number of keypoints and 3D points must be equal"
        assert self.S.shape[0] != 0, "The number of keypoints and 3D points must be greater than zero"
        assert self.S.shape[1] == 2, "The keypoints must be 2D"
        assert self.P.shape[1] == 3, "The 3D points must be 3D"

    def bootstrap(self, image_1: np.ndarray, image_2: np.ndarray) -> None:
        """
        Bootstraps the Vision Odometry object and generates the initial features and initial point cloud.

        Args:
            image_1 (numpy.ndarray): The first image.
            image_2 (numpy.ndarray): The second image.

        """
        self.prev_image = image_1

        # Find keypoints and descriptors of the images
        kp_des = self.get_image_keypoints_and_descriptors((image_1, image_2))
        kp_1, des_1 = kp_des[0]
        kp_2, des_2 = kp_des[1]
        # Indices of the inlier points
        kp_1_id = np.arange(len(kp_1))
        kp_2_id = np.arange(len(kp_2))

        T, points_3d_acceptable, matches_1_id, matches_2_id = self.get_relative_pose_and_3d_points_from_features(kp_1, des_1, kp_2, des_2)

        # Update the inlier points indices
        kp_1_id = kp_1_id[matches_1_id]
        kp_2_id = kp_2_id[matches_2_id]

        # Update state of the vision
        self.S = kp_1[kp_1_id]
        self.P = points_3d_acceptable
        self.scale = np.mean(np.linalg.norm(points_3d_acceptable, axis=1))

        assert self.S.shape[0] == self.P.shape[0], "The number of keypoints and 3D points must be equal"
        assert self.S.shape[0] != 0, "The number of keypoints and 3D points must be greater than zero"
        assert self.S.shape[1] == 2, "The keypoints must be 2D"
        assert self.P.shape[1] == 3, "The 3D points must be 3D"

    def get_image_keypoints_and_descriptors(self, images: Union[np.ndarray, Tuple[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the keypoints and descriptors of a set of images.

        Args:
            images (numpy.ndarray | tuple(numpy.ndarray)): The images.

        Returns:
            keypoints and descriptors (numpy.ndarray | tuple(numpy.ndarray)): The keypoints and descriptors of the images (Keypoint, Descriptor).
        """

        if not (isinstance(images, tuple) or isinstance(images, list)):
            images = (images,)

        output = []
        for image in images:
            keypoints, descriptors = self.feature_detector.detectAndCompute(image, None)
            keypoints = np.float32([keypoint.pt for keypoint in keypoints])  # Convert keypoints to numpy array
            output.append((keypoints, descriptors))

        return output if len(output) > 1 else output[0]  # Return a tuple if there is more than one image

    def get_relative_pose_and_3d_points_from_features(self, kp_1, des_1, kp_2, des_2):
        """
        Computes the relative pose of one frame with respect to the another using epipolar geometry and tringulates
        some 3D points from the set of 2D correspondences.

        Args:
            kp_1 (numpy.array) (N x 2): pixel coordinates of the keypoints in first image
            des_1 (numpy.array) (N x 128): descriptors of identified features in first image
            kp_2 (numpy.array) (N x 2): pixel coordinates of the keypoints in second image
            des_2 (numpy.array) (N x 128): descriptors of identified features in second image

        Returns:
            T_2 (numpy.array) (4 x 4): transformation matrix of second frame with respect to the first one
            points_3d_acceptable (numpy.array) (M x 2): filtered 3D points based on maximum distance threshold
            kp_1_id (numpy.array) (N x 2): indices of inliers keypoints for image 1
            kp_2_id (numpy.array) (N x 2): indices of inliers keypoints for image 2

        """

        # Indices of the inlier points
        kp_1_id = np.arange(len(kp_1))
        kp_2_id = np.arange(len(kp_2))

        # Find the matches between the two images
        matches_1_id, matches_2_id = self.find_2d_2d_matches(des_1, des_2)

        # Update the inlier points indices
        kp_1_id = kp_1_id[matches_1_id]
        kp_2_id = kp_2_id[matches_2_id]

        # Get the relative pose of the second image frame
        T_2, inlier_id_1, inlier_id_2 = self.get_pose_2d_2d(kp_1[kp_1_id], kp_2[kp_2_id])

        # Update the inlier points indices
        kp_1_id = kp_1_id[inlier_id_1]
        kp_2_id = kp_2_id[inlier_id_2]

        # Get the 3D points (in the first camera frame)
        T_1 = np.eye(4)  # The first pose is the origin
        points_3d = self.get_3d_points(kp_1[kp_1_id], kp_2[kp_2_id], T_1, T_2)

        # Remove points that are too far away or with negative depth
        MAX_DISTANCE = 300
        points_3d_distance = np.linalg.norm(points_3d, axis=1)
        acceptable_points_mask = (points_3d_distance < MAX_DISTANCE).astype(bool) & (points_3d[:, 2] > 0).astype(bool)
        # Update the inlier points indices
        kp_1_id = kp_1_id[acceptable_points_mask]
        kp_2_id = kp_2_id[acceptable_points_mask]
        points_3d_acceptable = points_3d[acceptable_points_mask]

        return T_2, points_3d_acceptable, kp_1_id, kp_2_id

    def find_2d_2d_matches(self, des_1: np.ndarray, des_2: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Matches the descriptors extracted from two images

        Args:
            des_1 (numpy.array) (N x 128): descriptors of identified features in first image
            des_2 (numpy.array) (N x 128): descriptors of identified features in second image

        Returns:
             Tuple of two numpy.ndarray:
                - matches_1_id (numpy.ndarray) (N, ): Indices of descriptors in the first image that matched with the second
                - matches_2_id (numpy.ndarray) (N, ): Indices of descriptors in the second image that matched with the first
        """

        # Match the features
        # matches = self.matcher.knnMatch(des_1, des_2, k=2)
        matches = self.matcher.match(des_1, des_2)

        good_matches = matches
        # Index of keypoints/descriptors of the matched features
        matches_1_id = np.array([m.queryIdx for m in good_matches], dtype=int)
        matches_2_id = np.array([m.trainIdx for m in good_matches], dtype=int)

        return matches_1_id, matches_2_id

    def get_pose_2d_2d(self, kp_1: np.ndarray, kp_2: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Computes the pose of the second camera frame with respect to the first camera frame using the 5 point RANSAC algorithm
        for the estimation of the essential matrix

        Args:
            q1 (numpy.ndarray) (Nx2): The image points of the matched features in the first image.
            q2 (numpy.ndarray) (Nx2): The image points of the matched features in the second image.

        Returns:
            T (numpy.ndarray) (4x4): The homogeneous transformation matrix of the second pose.

        """
        assert kp_1.shape == kp_2.shape, "The number of points must be equal"
        assert kp_1.shape[1] == 2, "The points must be 2D"
        assert kp_1.shape[0] > 8, "The number of points must be greater than 8"  # TODO: Set a minimum number of points

        K = self.K

        # Indices of the inlier points
        kp_1_id = np.arange(len(kp_1))
        kp_2_id = np.arange(len(kp_2))

        # Find the essential matrix
        p = 0.999
        err = 1.0
        if self.dataset == 2:
            p = 0.99
            err = 5.0
        E, mask_E = cv2.findEssentialMat(kp_1, kp_2, K, cv2.RANSAC, p, err)

        # Update the inlier points indices
        kp_1_id = kp_1_id[mask_E.ravel() == 1]
        kp_2_id = kp_2_id[mask_E.ravel() == 1]

        # Get the correct pose
        success, R, t, mask_P = cv2.recoverPose(E, kp_1[kp_1_id], kp_2[kp_2_id])

        # t is defined as the translation on the points and not on the camera, this means that if the camera moves forward
        # the points are moving backward. For this reason -t is used instead of t
        t = - t

        # Update the inlier points indices
        kp_1_id = kp_1_id[mask_P.ravel() == 255]  # 255 is the value of the inlier mask (Ask OpenCV why)
        kp_2_id = kp_2_id[mask_P.ravel() == 255]

        # Build the transformation matrix
        T_2 = self.RT_to_pose(R, t)  # Relative pose of the second camera frame with respect to the first camera frame

        return T_2, kp_1_id, kp_2_id


    ## Continuous operation
    def process_new_frame(self, image: np.ndarray):
        """
        Processes a new frames by tracking features, estimating the motion, triangulating new 3D points and adding new
        candidate keypoints.

        Args:
            image (numpy.ndarray): The new image.

        Returns:
            T (numpy.array) (4 x 4): Transformation matrix of current frame with respect to world frame
            kp_prev (numpy.array) (N x 2): Pixel coordinates of keypoints in the previous image (needed for plotting)

        """

        ## Track new keypoints from old keypoints using KLT
        new_kp, mask, err = cv2.calcOpticalFlowPyrLK(self.prev_image, image, self.S, None, **self.klt_params)
        # Filter the keypoints
        mask = mask.ravel() == 1

        kp_prev = self.S[mask] # Needed later for plotting

        # Update state
        self.S = new_kp[mask]
        self.P = self.P[mask]

        # Find pose with PnP
        T, inliers = self.get_pose_3d_2d(self.P, self.S)
        # print(f'Number of inliers: {np.sum(mask) / len(mask)}') # Consistenly higher than 90 %

        # Update keypoints with inliers
        kp_prev = kp_prev[inliers]
        self.S = self.S[inliers]
        self.P = self.P[inliers]

        T = self.pose_refinement_3d_2d(T, self.P, self.S)


        ## Track candidate keypoints from old candidate keypoints using KLT
        if len(self.C) != 0:
            new_kp, mask, err = cv2.calcOpticalFlowPyrLK(self.prev_image, image, self.C, None, **self.klt_params)

            # Filter the candidates
            mask = mask.ravel() == 1  # Mask of the candidates that were tracked
            self.C = new_kp[mask]  # Update the candidate keypoints
            self.F = self.F[mask]
            self.T = self.T[mask]

            # Check which candidates could be added to the keypoints
            # Triangulate the points and check the baseline angle
            points_3d, angle_mask = self.check_candidate_keypoints(T, self.base_line_angle)

            # Update keypoints with the new candidates
            self.S = np.concatenate((self.S, self.C[angle_mask]), axis=0)
            self.P = np.concatenate((self.P, points_3d), axis=0)

            # For plotting
            self.added_kp = len(self.C[angle_mask])

            # Remove the candidates that were added to the keypoints
            self.C = self.C[~angle_mask]
            self.F = self.F[~angle_mask]
            self.T = self.T[~angle_mask]

        ## Find new candidate keypoints
        # Initialize new candidate keypoints with Harris
        new_corners = cv2.goodFeaturesToTrack(image, 1500, 0.05, 10).reshape(-1,2)
        # print('Harris:', new_corners.shape)
        # Filter new candidate keypoints that are too close to previous keypoints
        previous_keypoints = np.concatenate((self.S, self.C), axis=0)
        new_corners = self.filter_new_keypoints(previous_keypoints, new_corners, 1)

        # Update candidate keypoints
        self.C = np.concatenate((self.C, new_corners), axis=0)
        self.F = np.concatenate((self.F, new_corners), axis=0)
        self.T = np.concatenate((self.T, np.tile(T, (new_corners.shape[0], 1, 1))), axis=0)

        self.prev_image = image

        # Get R and t for plotting
        R = T[:3, :3].T
        t = -T[:3, :3].T @ T[:3, 3]

        return R, t

    def check_candidate_keypoints(self, T: np.ndarray, baseline_angle):
        """
        Tringulates new 3D points in front of the camera and filters them to make sure that the baseline angle is above
        a given threshold

        Args:
            T (numpy.ndarray) (4x4): The camera poses of the current observation of the candidate keypoints.

        Returns:
            mask (numpy.ndarray) (N, ): The mask of the candidate keypoints that can be added to the keypoints.
        """
        # Triangulate the points
        points_3d = np.zeros((len(self.C), 3))

        for i in range(len(self.C)):
            point_3d = self.get_3d_points(self.F[[i]], self.C[[i]], self.T[i], T)
            points_3d[i] = point_3d

        # Indices of points for mask
        idx = np.arange(len(points_3d))

        # Only take the 3D points traingulated in front of the camera
        points_3d_W_hom = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        points_3d_C = np.dot(T, points_3d_W_hom.T).T
        points_3d_C = points_3d_C[:, :-1]
        positive_depth_mask = points_3d_C[:, 2] > 0
        idx = idx[positive_depth_mask]
        points_3d_C = points_3d_C[positive_depth_mask]

        # Rescaling
        current_scale = np.mean(np.linalg.norm(points_3d_C, axis=1))
        points_3d_C *= self.scale / current_scale


        # Remove points too far away
        max_distance_mask = np.linalg.norm(points_3d_C, axis=1) < 300
        idx = idx[max_distance_mask]
        points_3d_C = points_3d_C[max_distance_mask]

        # # Transform points to world frame
        # points_3d_C_hom = np.hstack([points_3d_C, np.ones((len(points_3d_C), 1))])
        # T_wc = self.RT_to_pose(T[:3,:3].T, -T[:3,:3].T @ T[:3,3])
        # points_3d = np.dot(T_wc, points_3d_C_hom.T).T
        # points_3d = points_3d[:,:-1]
        #
        # # Baseline angle filtering
        # vector_1 = points_3d - T[:3, 3]
        # vector_2 = points_3d - self.T[idx, :3, 3]
        # angle = np.arccos(np.sum(vector_1 * vector_2, axis=1) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2)))
        # angle_mask = np.abs(angle) > baseline_angle

        # Baseline angle filtering
        vector_1 = points_3d[idx] - T[:3, 3]
        vector_2 = points_3d[idx] - self.T[idx, :3, 3]
        angle = np.arccos(np.sum(vector_1 * vector_2, axis=1) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2)))
        angle_mask = np.abs(angle) > baseline_angle

        # Definition of joint mask
        idx = idx[angle_mask]
        mask = np.zeros(len(self.C)).astype(bool)
        mask[idx] = True

        # print(f'Rejected points: {(len(mask) - np.sum(mask)) * 100 /  len(mask)} %')

        points_3d = points_3d[mask]
        # points_3d = points_3d[angle_mask]

        return points_3d, mask

    def filter_new_keypoints(self, prev_keypoints, curr_keypoints, distance_threshold=1.0):
        """
        Filters out new keypoints that are too close to previous keypoints.

        Args:
            prev_keypoints (numpy.array) (N x 2): Numpy array of previous keypoints.
            curr_keypoints (numpy.array) (N x 2): Numpy array of current keypoints.
            distance_threshold: Distance threshold below which a new keypoint will be removed.
        Returns:
            out (numpy.array) (M, 2): Filtered new keypoints.
        """
        # Compute the pairwise distances between all previous and current keypoints
        distances = np.linalg.norm(prev_keypoints[:, np.newaxis, :] - curr_keypoints, axis=2)

        # Find the minimum distance for each current keypoint
        min_distances = np.min(distances, axis=0)

        # Keep only the keypoints where the minimum distance is greater than the threshold
        mask = min_distances > distance_threshold

        out = curr_keypoints[mask]

        return out

    def get_pose_3d_2d(self, points_3d: np.ndarray, kp: np.ndarray):
        """
        Computes the pose of the current frame with respect to the world coordinate system by using P3P RANSAC

        Args:
            points_3d (numpy.array) (N x 3): Coordinates of the 3D points
            kp (numpy.array) (N x 2): Pixel coordinates of corresponding 3D points

        Returns:
            T (numpy.array) (4 x 4): Transformation matrix of the current frame with respect to the world frame
            inliers (numpy.array) (N, ): Mask of inliers

        """
        assert points_3d.shape[0] != 0, "The number of points must be greater than zero"
        assert kp.shape[0] == points_3d.shape[0], "The number of points must be equal"
        assert kp.shape[0] >= 3, "The number of points must be greater than 8"  # TODO: Set a minimum number of points

        K = self.K
        # TODO: Understand function parameters

        # repr_err = self.pnp_ransac_params['reprojectionError']
        # success, R_vec, t, inliers = cv2.solvePnPRansac(points_3d, kp, K, **self.pnp_ransac_params)
        # while not success:
        #     self.pnp_ransac_params['reprojectionError'] += 0.25
        #     success, R_vec, t, inliers = cv2.solvePnPRansac(points_3d, kp, K, **self.pnp_ransac_params)
        #
        # self.pnp_ransac_params['reprojectionError'] = repr_err

        success, R_vec, t, inliers = cv2.solvePnPRansac(points_3d, kp, K, **self.pnp_ransac_params)

        R = cv2.Rodrigues(R_vec)[0]
        T = self.RT_to_pose(R, t)

        return T, inliers.ravel()

    def get_3d_points(self, kp_1: np.ndarray, kp_2: np.ndarray, T_1: np.ndarray, T_2: np.ndarray) -> np.ndarray:

        """
        Computes the 3D points (defined in reference frame 1) from the matched points q1 and q2 and the homogeneous transformation matrix T of 
        the second pose (in which the q2 points are defined).

        Args:
            q1 (numpy.ndarray) (Nx2): The image points of the matched features in the first image.
            q2 (numpy.ndarray) (Nx2): The image points of the matched features in the second image.
            T (numpy.ndarray) (4x4): The homogeneous transformation matrix of the second pose (in which the q2 points are defined).

        Returns:
            points_3d (numpy.ndarray) (Nx3): The 3D points in the first camera frame.

        """
        assert kp_1.shape == kp_2.shape, "The number of dimensions must be equal"
        assert kp_1.shape[1] == 2, "The points must be 2D"
        assert kp_1.shape[0] != 0, "The number of points must be greater than zero"

        K = self.K

        # Build the projection matrices
        P1 = K @ T_1[:3, :]
        P2 = K @ T_2[:3, :]

        # Triangulate the points
        points_4d_1 = cv2.triangulatePoints(P1, P2, kp_1.T, kp_2.T)  # Homogeneous coordinates in the first camera frame

        # Convert to un-homogeneous coordinates
        points_4d_1 = points_4d_1 / points_4d_1[3]

        points_3d = points_4d_1[:3].T

        return points_3d

    def pose_refinement_3d_2d(self, T_i: np.ndarray, points_3d: np.ndarray, kp: np.ndarray) -> np.ndarray:

        """
        Refines the pose using by minimizing the reprojection error through Levenberg-Marquardt.

        Args:
            T_i (numpy.array) (4 x 4): Transformation matrix of current frame
            points_3d (numpy.array) (N x 3): Coordinates of 3D points
            kp (numpy.array) (N x 2): Pixel coordinates of corresponding 2D points

        Returns:
            T (numpy.array) (4 x 4): Refined transformation matrix of current frame
        """

        assert kp.shape[0] == points_3d.shape[0], "The number of points must be equal"
        assert kp.shape[1] == 2, "The points must be 2D"
        assert points_3d.shape[1] == 3, "The 3D points must be 3D"
        assert kp.shape[0] != 0, "The number of points must be greater than zero"

        K = self.K

        # Refine the pose using non-linear least squares
        func = self._get_squared_reprojection_error_func_3d_2d(points_3d, kp)

        # Format the initial guess as a 1-D array
        R, t = self.pose_to_RT(T_i)
        r_vec = cv2.Rodrigues(R)[0]  # Convert the rotation matrix to a rotation vector

        x_0 = np.concatenate((r_vec.ravel(), t.ravel()))

        # Run the optimization (Levenberg-Marquardt method)
        res = least_squares(func, x_0, method='trf')

        # Build the final transformation matrix
        R = cv2.Rodrigues(res.x[:3])[0]
        t = res.x[3:]

        T = self.RT_to_pose(R, t)

        return T

    def _get_squared_reprojection_error_func_3d_2d(self, points_3d, kp):
        """
        Returns the function used in pose_refinement_3d_2d for refining the pose

        Args:
            points_3d (numpy.array) (N x 3): Coordinates of 3D points
            kp (numpy.array) (N x 2): Pixel coordinates of corresponding 2D points

        Returns:
            fun (function): Function of the pose x

        """
        K = self.K

        def fun(x):
            # Build T from x
            T = np.concatenate((cv2.Rodrigues(x[:3])[0], x[3:].reshape(-1, 1)), axis=1)
            T = np.concatenate((T, np.array([0, 0, 0, 1]).reshape(1, -1)), axis=0)

            # Project the 3d points to the image plane
            points_projected = cv2.projectPoints(points_3d, T[:3, :3], T[:3, 3], K, None)[0].reshape(-1, 2)

            # Calculate the reprojection error (distance between the projected points and the actual points)
            error = np.linalg.norm(points_projected - kp, axis=1)

            # Create 1-D residual vector
            residual = error.ravel()

            return residual

        return fun


    ## Plotting
    def setup_plot_all(self, img: np.ndarray):

        # Close all existing figures
        plt.close('all')
        # Create a figure with 4 subplots arranged in a 2x5 grid
        fig = plt.figure(figsize=(16, 7))

        # Upper text
        # Add main title to the figure
        fig.suptitle('Monocular Visual Odometry Pipeline', fontsize=20, fontweight='bold', x=0.5, y=0.95)
        if self.dataset == 0:
            fig.text(0.5, 0.88, 'Kitti dataset', ha='center', fontsize=14, fontstyle='italic')
        elif self.dataset == 1:
            fig.text(0.5, 0.88, 'Malaga dataset', ha='center', fontsize=14, fontstyle='italic')
        elif self.dataset == 2:
            fig.text(0.5, 0.88, 'Parking dataset', ha='center', fontsize=14, fontstyle='italic')

        intructions_text = fig.text(0.5, 0.03, 'Press ''c'' to continue', ha='center', fontsize=14, fontstyle='italic')

        # Define a finer grid size for more control
        grid_size = (10, 24)  # This is just an example; adjust as needed for your layout

        # Plot 0: Image with keypoints
        if self.dataset == 0:
            ax0 = plt.subplot2grid(grid_size, (1, 0), colspan=7, rowspan=4)
        elif self.dataset == 1:
            ax0 = plt.subplot2grid(grid_size, (0, 0), colspan=7, rowspan=4)
        elif self.dataset == 2:
            ax0 = plt.subplot2grid(grid_size, (0, 0), colspan=7, rowspan=4)

        scat_old_kp = ax0.scatter([], [], label='Old keypoints', color='red', s=3)
        scat_new_kp = ax0.scatter([], [], label='New keypoints', color='blue', s=3)
        image_ax = ax0.imshow(img)  # Placeholder image (img dimension)
        if self.dataset == 0:
            ax0.legend(loc='lower center', bbox_to_anchor=(0.5, -0.46), fancybox=True, shadow=True, edgecolor='black')
        elif self.dataset == 1:
            ax0.legend(loc='lower center', bbox_to_anchor=(0.5, -0.27), fancybox=True, shadow=True, edgecolor='black')
        elif self.dataset == 2:
            ax0.legend(loc='lower center', bbox_to_anchor=(0.5, -0.27), fancybox=True, shadow=True, edgecolor='black')

        ax0.set_xticks([])  # Remove x-axis ticks
        ax0.set_yticks([])  # Remove y-axis ticks

        # Plot 1: Keypoints statistics
        # Place it directly below ax0 and make it the same width
        ax1 = plt.subplot2grid(grid_size, (6, 0), colspan=7, rowspan=3)
        line_kp, = ax1.plot([], [], label='Number of keypoints', color='blue')
        line_candidates, = ax1.plot([], [], label='Number of candidate keypoints', color='red')
        line_new_kp, = ax1.plot([], [], label='New keypoints', color='green')
        ax1.title.set_text('Keypoint Statistics')
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.62), fancybox=True, shadow=True, edgecolor='black')

        # Plot 2: Global trajectory
        ax2 = plt.subplot2grid(grid_size, (1, 8), colspan=8, rowspan=9)
        scat_trajectory = ax2.scatter([], [], label='Estimated Trajectory', color='green', s=10)
        ax2.title.set_text('Estimated Global Trajectory')
        # Fixed limits based on dataset
        if self.dataset == 0:
            ax2.set_xlim([-600, 600])
            ax2.set_ylim([-500, 1000])
        elif self.dataset == 1:
            ax2.set_xlim([-80, 10])
            ax2.set_ylim([-10, 70])
        elif self.dataset == 2:
            ax2.set_xlim([0, 40])
            ax2.set_ylim([-20, 20])

        # Plot 3: Local trajectory with 3D points
        ax3 = plt.subplot2grid(grid_size, (1, 17), colspan=6, rowspan=9)
        scat_old_points = ax3.scatter([], [], label='Old 3D points', color='gray', s=3)
        scat_new_points = ax3.scatter([], [], label='New 3D points', color='blue', s=7, marker='*')
        scat_local_trajectory = ax3.scatter([], [], color='red', s=12)
        ax3.title.set_text(f'Local Trajectory over last {self.n_last_frames} frames')
        ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.11), fancybox=True, shadow=True, edgecolor='black')
        ax3.set_xticks([])  # Remove x-axis ticks
        ax3.set_yticks([])  # Remove y-axis ticks

        # Adjust the spacing of the subplots
        plt.subplots_adjust(left=0.04, right=0.97, wspace=0.2, hspace=0.2)

        # Output data
        plot0 = (ax0, scat_old_kp, scat_new_kp, image_ax)
        plot1 = (ax1, line_kp, line_candidates, line_new_kp)
        plot2 = (ax2, scat_trajectory)
        plot3 = (ax3, scat_local_trajectory, scat_new_points, scat_old_points)
        plots = (plot0, plot1, plot2, plot3, intructions_text)

        return fig, plots

    def update_plot(self, fig, plots, data, img_index):
        plot0, plot1, plot2, plot3, _ = plots
        data0, data1, data2, data3 = data

        # Plot 0: Image with keypoints
        self.plot_img_with_keypoints(plot0, data0, img_index)

        # Plot 1: Keypoint statistics
        self.plot_keypoints_statistics(plot1, data1, img_index)

        # Plot 2: Global trajectory
        self.plot_global_trajectory(plot2, data2)

        # Plot 3: Local trajectory with 3D points
        self.plot_local_trajectory(plot3, data3, img_index)

        # Refresh the plot
        fig.canvas.draw_idle()
        plt.pause(0.05)  # Short pause to update the plot

    def plot_img_with_keypoints(self, plot0, data0, img_index):
        # Get plotters
        ax0, scat_old_kp, scat_new_kp, image_ax = plot0
        # Get data
        old_kp, new_kp, img = data0

        # Update subplot
        ax0.title.set_text(f'Tracked Keypoints\nImage {img_index+1}')
        image_ax.set_data(img)
        if old_kp is not None:
            scat_old_kp.set_offsets(old_kp)
        scat_new_kp.set_offsets(new_kp)

    def plot_keypoints_statistics(self, plot1, data1, img_index):
        # Get plotters
        ax1, line_kp, line_candidates, line_new_kp = plot1
        # Get data
        n_kp, n_candidate_kp, n_added_kp = data1

        if img_index != 0:
            line_kp.set_data(range(0, img_index+1), n_kp)
            line_candidates.set_data(range(0, img_index+1), n_candidate_kp)
            line_new_kp.set_data(range(0, img_index+1), n_added_kp)

            # Update the x-axis limits to focus from min_index to img_index
            ax1.set_xlim(0, img_index)

            # adjust the y-axis to fit the data
            all_data = np.vstack((n_kp[:img_index], n_candidate_kp[:img_index], n_added_kp[0:img_index])).T
            ax1.set_ylim(all_data.min(), all_data.max()+50)

    def plot_global_trajectory(self, plot2, data2):
        # Get plotters
        ax2, scat_trajectory = plot2
        # Get data
        trajectory = data2

        xz_data = np.vstack([trajectory[:,0], trajectory[:,2]]).T
        scat_trajectory.set_offsets(xz_data)

    def plot_local_trajectory(self, plot3, data3, img_index):
        # Get plotters
        ax3, scat_local_trajectory, scat_new_points, scat_old_points = plot3
        # Get data
        local_trajectory, R, new_points_3d, old_points_3d = data3

        # Downscale everything to make all points visible
        if self.dataset == 0:
            scale = 0.4
        elif self.dataset == 1:
            scale = 1.5
        elif self.dataset == 2:
            scale = 1.5

        T_vec_inv = scale * R.T @ local_trajectory.T
        x_R_coords = T_vec_inv.T[:, 0]
        z_R_coords = T_vec_inv.T[:, 2]
        xz_data = np.vstack((x_R_coords, z_R_coords)).T
        scat_local_trajectory.set_offsets(xz_data)

        new_points_3d_inv = scale *R.T @ new_points_3d.T
        xz_new_points_inv = new_points_3d_inv.T[:, [0, 2]]
        scat_new_points.set_offsets(xz_new_points_inv)

        if old_points_3d is not None:
            old_points_3d_inv = scale * R.T @ old_points_3d.T
            xz_old_points_inv = old_points_3d_inv.T[:, [0, 2]]
            scat_old_points.set_offsets(xz_old_points_inv)


        ax3.set_xlim(x_R_coords[-1] - 8, x_R_coords[-1] + 8)
        ax3.set_ylim(z_R_coords[-1] - 5, z_R_coords[-1] + 30)
        ax3.xaxis.set_visible(False)
        ax3.yaxis.set_visible(False)

    def plot_final_trajectory_ground_truth(self, trajectory, ground_truth):
        fig, axs = plt.subplots(1, 2, figsize=(8, 6))

        fig.suptitle('Comparison between estimated trajectory and ground truth', fontsize=20, fontweight='bold', x=0.5, y=0.95)
        if self.dataset == 0:
            fig.text(0.5, 0.88, 'Kitti dataset', ha='center', fontsize=14, fontstyle='italic')
        elif self.dataset == 2:
            fig.text(0.5, 0.88, 'Parking dataset', ha='center', fontsize=14, fontstyle='italic')

        axs[0].set_title('Estimated trajectory')
        axs[0].plot(trajectory[:, 0], trajectory[:, 2])
        axs[0].scatter(trajectory[:, 0], trajectory[:, 2], color='green', s=10)
        axs[0].axis('equal')

        # Right subplot (index 1)
        T = ground_truth[:, :3, 3]

        axs[1].set_title('Ground truth')
        axs[1].plot(T[:, 0], T[:, 2])
        axs[1].scatter(T[:, 0], T[:, 2], color='red', s=10)
        axs[1].axis('equal')

        fig.subplots_adjust(bottom=0.1, top=0.8)  # Adjust the bottom and top

    def plot_final_trajectory_without_ground_truth(self, trajectory):
        fig = plt.figure(figsize=(8, 6))
        fig.suptitle('Estimated trajectory', fontsize=20, fontweight='bold', x=0.5, y=0.95)
        fig.text(0.5, 0.88, 'Malaga dataset', ha='center', fontsize=14, fontstyle='italic')

        # Creating a subplot with adjustable parameters
        ax = fig.add_subplot(111)
        ax.plot(trajectory[:, 0], trajectory[:, 2])
        ax.scatter(trajectory[:, 0], trajectory[:, 2], color='green', s=10)
        ax.axis('equal')
        fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)

    ## Utility Functions
    def RT_to_pose(self, R, t):
        """
        Builds the homogeneous transformation matrix from the rotation matrix and translation vector.
        """
        return np.concatenate((np.concatenate((R, t.reshape(-1, 1)), axis=1), np.array([0, 0, 0, 1]).reshape(1, -1)),axis=0)

    def pose_to_RT(self, T):
        """
        Extracts the rotation matrix and translation vector from the homogeneous transformation matrix.
        """
        return T[:3, :3], T[:3, 3]
