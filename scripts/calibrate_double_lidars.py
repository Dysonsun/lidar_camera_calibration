#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Modified: hsh (sunskyhsh@hotmail.com)
Version : 1.2.1
Date    : Sep 7, 2020

Description:
Script to find the transformation between the Camera and the LiDAR

Example Usage:
1. To perform calibration using the GUI to pick correspondences chessboard points:

    $ rosrun lidar_camera_calibration calibrate_camera_lidar.py --calibrate \
        --image_topic XXX --pointcloud_topic XXX [--camera_lidar_topic XXX \
        --chessboard_size 7x9 --square_size 0.12 --select_lidar_points 500]
        
    The corresponding images and pointclouds will be save as following:
    - PKG_PATH/calibration_data/lidar_camera_calibration/[camera_name]_pcd/[0...n].jpg
    - PKG_PATH/calibration_data/lidar_camera_calibration/[camera_name]_pcd/[0...n].pcd

    The calibrate extrinsic are saved as following:
    - PKG_PATH/calibration_data/lidar_camera_calibration/extrinsics.yaml
    --> 'R'     : rotation matrix (3, 3)
    --> 'T'     : translation offsets (3, )
    --> 'euler' : euler angles (3, )

2. To display the LiDAR points projected on to the camera plane:

    $ rosrun lidar_camera_calibration calibrate_camera_lidar.py --project \
        --image_topic XXX --pointcloud_topic XXX [--camera_lidar_topic XXX]

Notes:
Make sure this file has executable permissions:
$ chmod +x calibrate_camera_lidar.py

Reference:
Huang, Lili, and Matthew Barth. "A novel multi-planar LIDAR and computer vision calibration procedure using 2D patterns for automated navigation." 2009 IEEE Intelligent Vehicles Symposium. IEEE, 2009.
'''

# Python 2/3 compatibility
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# Built-in modules
import os
import sys
import time
import threading
import multiprocessing
import math
import six

# External modules
import numpy as np
import matplotlib.cm
import yaml
import argparse
from scipy import optimize
from sklearn.decomposition import PCA

# ROS modules
PKG = 'lidar_camera_calibration'
import roslib; roslib.load_manifest(PKG)
import rosbag
import rospy
import ros_numpy
import message_filters
# from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import PointCloud2
import pptk
import pcl

# local modules
from checkerboard import detect_checkerboard
from utils import *

# Global variables
PAUSE = False
FIRST_TIME = True
KEY_LOCK = threading.Lock()
TF_BUFFER = None
TF_LISTENER = None
WATCH_LIVESTREAM_OUTCOME = False
REMOVE_LAST_FRAME = False

# Global paths
PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CALIB_PATH = 'calibration_data/double_lidar_calibration'


'''
Keyboard handler thread
Inputs: None
Outputs: None
'''
def handle_keyboard():
    global KEY_LOCK, PAUSE, WATCH_LIVESTREAM_OUTCOME, REMOVE_LAST_FRAME
    key = six.moves.input('Press [ENTER] to pause and pick points or \n[l] to watch livestream outcome or \n[c] to cancel livestream mode\n[r] to remove last frame\n')
    if (key == 'l'):
        print("[l] pressed. watch livestream outcome")
        with KEY_LOCK:
            PAUSE = True
            WATCH_LIVESTREAM_OUTCOME = True
    elif (key == 'c'):
        print("[c] pressed. cancel livestream mode")
        with KEY_LOCK:
            PAUSE = True
            WATCH_LIVESTREAM_OUTCOME = False
    elif (key == 'r'):
        print("[r] pressed. remove last frame")
        with KEY_LOCK:
            REMOVE_LAST_FRAME = True
    else:
        print("[ENTER] pressed")
        with KEY_LOCK: PAUSE = True


'''
Start the keyboard handler thread
Inputs: None
Outputs: None
'''
def start_keyboard_handler():
    keyboard_t = threading.Thread(target=handle_keyboard)
    keyboard_t.daemon = True
    keyboard_t.start()


class DoubleLiDARCalibrator:
    '''
    The main ROS node which handles the topics

    Inputs:
        camera_info - [str] - ROS sensor camera info topic
        image_color - [str] - ROS sensor image topic
        velodyne - [str] - ROS velodyne PCL2 topic
        camera_lidar - [str] - ROS projected points image topic

    Outputs: None
    '''
    def __init__(self, args, pointcloud_topic, camera_lidar_topic=None):
        self.args = args
        # Start node
        rospy.init_node('calibrate_camera_lidar', anonymous=True)
        rospy.loginfo('Current PID: [%d]' % os.getpid())
        rospy.loginfo('Projection mode: %s' % PROJECT_MODE)
        rospy.loginfo('PointCloud2 topic:')
        print(pointcloud_topic)
        # rospy.loginfo('Output topic: %s' % camera_lidar_topic)
        assert len(pointcloud_topic) == 2
        # self.calibrate_mode = self.args.calibrate
        assert len(args.lidar_name) == 2
        self.lidar_name = args.lidar_name
        assert ('x' in args.chessboard_size)
        self.chessboard_size = tuple([int(x) for x in args.chessboard_size.split('x')])
        assert (len(self.chessboard_size) == 2)
        self.chessboard_diagonal = np.sqrt(self.chessboard_size[0]**2+self.chessboard_size[1]**2) * args.square_size
        # self.camera_matrix = None
        # self.image_shape = None
        self.master_lidar_points = []
        self.master_centroids = []
        self.slave_lidar_points = []
        self.master_normal_vector = []
        self.frame_count = 0
        self.use_L1_error = args.use_L1_error
        self.calibrate_min_frames = args.calibrate_min_frames
        self.chessboard_ROI_points = []
        self.procs = []
        self._threads = []
        self._lock = threading.Lock()
        self.calibration_finished = False
        self.slave_to_master_R = None
        self.slave_to_master_T = None
        self.folder = os.path.join(PKG_PATH, CALIB_PATH)
        if (not os.path.exists(self.folder)):
            os.mkdir(self.folder)
        self.success = False
        self.lidar_points_num = 0
        self.callback_running = False
        if (not PROJECT_MODE):
            rospy.loginfo("use_L1_error: %s" % str(self.use_L1_error))
        if (args.transform_pointcloud):
            self.lidar_rotation_matrix = np.zeros((3,3))
            # extrinsics = readYAMLFile(os.path.join(PKG_PATH, 'calibration_data/lidar_calibration', self.lidar_name[0]+'_extrinsics.yaml'))
            # rotation_matrix = np.array(extrinsics['R']['data']).reshape(extrinsics['R']['rows'], extrinsics['R']['cols'])
            # translation_vector = np.array(extrinsics['T']['data']).reshape(extrinsics['T']['rows'], extrinsics['T']['cols'])
            # self.lidar_translation_vector = np.array(extrinsics['T']['data']).reshape(extrinsics['T']['rows'], extrinsics['T']['cols'])
            # euler_angles = np.array(extrinsics['euler']['data']).reshape(extrinsics['euler']['rows'], extrinsics['euler']['cols'])
            # # reverse_rotation_matrix = rotation_matrix_from_vectors(np.array([0,0,1]), normal_vector)
            # reverse_rotation_matrix = np.linalg.inv(rotation_matrix)
            # reverse_euler_angles = euler_from_matrix(reverse_rotation_matrix)

            # alfa, beta, gama = reverse_euler_angles
            # alfa, beta, gama = np.array([-0.9243, 0.3538, 0.0057]) * np.pi / 180.
            # alfa = -alfa
            alfa, beta, gama = np.array([-0.866996, 0.803254, -0.006078]) * np.pi / 180.
            self.lidar_rotation_matrix[0,0] = np.cos(beta)*np.cos(gama) - np.sin(beta)*np.sin(alfa)*np.sin(gama)
            self.lidar_rotation_matrix[1,0] = np.cos(beta)*np.sin(gama) + np.sin(beta)*np.sin(alfa)*np.cos(gama)
            self.lidar_rotation_matrix[2,0] = -np.cos(alfa)*np.sin(beta)

            self.lidar_rotation_matrix[0,1] = -np.cos(alfa)*np.sin(gama)
            self.lidar_rotation_matrix[1,1] = np.cos(alfa)*np.cos(gama)
            self.lidar_rotation_matrix[2,1] = np.sin(alfa)

            self.lidar_rotation_matrix[0,2] = np.sin(beta)*np.cos(gama) + np.cos(beta)*np.sin(alfa)*np.sin(gama)
            self.lidar_rotation_matrix[1,2] = np.sin(beta)*np.sin(gama) - np.cos(beta)*np.sin(alfa)*np.cos(gama)
            self.lidar_rotation_matrix[2,2] = np.cos(alfa)*np.cos(beta)

            # self.lidar_rotation_matrix[3,0] = 0
            # self.lidar_rotation_matrix[3,1] = 0
            # self.lidar_rotation_matrix[3,2] = 0
            # self.lidar_rotation_matrix[3,3] = 1.0
            self.lidar_translation_vector = np.zeros((3,1))
            self.lidar_translation_vector[0] = 0
            self.lidar_translation_vector[1] = 1.28
            self.lidar_translation_vector[2] = 1.8

        # Subscribe to topics
        master_pointcloud_sub = message_filters.Subscriber(pointcloud_topic[0], PointCloud2)
        slave_pointcloud_sub = message_filters.Subscriber(pointcloud_topic[1], PointCloud2)

        # Publish output topic
        # image_pub = None
        # if camera_lidar_topic: image_pub = rospy.Publisher(camera_lidar_topic, Image, queue_size=5)

        # Synchronize the topics by time
        self.ats = message_filters.ApproximateTimeSynchronizer(
            [master_pointcloud_sub, slave_pointcloud_sub], queue_size=1, slop=0.1)
        self.ats.registerCallback(self.lidar_callback)

        # Start keyboard handler thread
        if not args.project: start_keyboard_handler()

        # Keep python from exiting until this node is stopped
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo('Shutting down')

    '''
    Runs the LiDAR point selection GUI process

    Inputs:
        velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
        now - [int] - ROS bag time in seconds

    Outputs:
        Picked points saved in PKG_PATH/CALIB_PATH/pcl_corners.npy
    '''
    def extract_lidar_points(self, pointcloud_msg, is_master_lidar=True, proc_results=None):
        # Log PID
        rospy.loginfo('3D Picker PID: [%d]' % os.getpid())

        # Extract points data
        points = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud_msg)
        points = np.asarray(points.tolist())
        if (len(points.shape)>2):
            points = points.reshape(-1, points.shape[-1])
        points = points[~np.isnan(points).any(axis=1), :]


        if (is_master_lidar):
            if (self.args.transform_pointcloud):
                points = (self.lidar_rotation_matrix.dot(points[:,:3].T) + self.lidar_translation_vector).T
            points2pcd(points, os.path.join(PKG_PATH, CALIB_PATH, '_'.join(self.lidar_name), str(self.lidar_points_num)+"_0.pcd"))
        else:
            points2pcd(points, os.path.join(PKG_PATH, CALIB_PATH, '_'.join(self.lidar_name), str(self.lidar_points_num)+"_1.pcd"))

        # Select points within chessboard range
        # if 'left' in self.lidar_name[1]:
        #     inrange = np.where((points[:, 0] < -3) &
        #                     (points[:, 0] > -10) &
        #                     (points[:, 1] < 5) &
        #                     (points[:, 1] > -5) &
        #                     (points[:, 2] < 4) &
        #                     (points[:, 2] > 0.2))
        # elif 'right' in self.lidar_name[1]:
        #     inrange = np.where((points[:, 0] > 3) &
        #                     (points[:, 0] < 10) &
        #                     (points[:, 1] < 5) &
        #                     (points[:, 1] > -5) &
        #                     (points[:, 2] < 4) &
        #                     (points[:, 2] > 0.2))
        # else:
        #     inrange = np.where((np.abs(points[:, 0]) < 7) &
        #                     (points[:, 1] < 25) &
        #                     (points[:, 1] > 0) &
        #                     (points[:, 2] < 4) &
        #                     (points[:, 2] > 0.2))
        inrange = np.where((np.abs(points[:, 0]) < 7) &
                        (points[:, 1] < 25) &
                        (points[:, 1] > 0) &
                        (points[:, 2] < 4) &
                        (points[:, 2] > 0.2))
        # points = points[inrange[0]]
        if points.shape[0] < 10:
            rospy.logwarn('Too few PCL points available in range')
            return None, None, None

        # pptk viewer
        viewer = pptk.viewer(points[:, :3])
        # if 'left' in self.lidar_name[1]:
        #     viewer.set(lookat=(0,2,4))
        #     viewer.set(phi=0)
        # elif 'right' in self.lidar_name[1]:
        #     viewer.set(lookat=(0,2,4))
        #     viewer.set(phi=-3.3)
        # else:
        #     viewer.set(lookat=(0,0,4))
        #     viewer.set(phi=-1.57)
        # viewer.set(lookat=(0,0,4))
        # viewer.set(phi=-1.57)
        # viewer.set(theta=0.4)
        viewer.set(r=4)
        viewer.set(floor_level=0)
        viewer.set(point_size=0.02)
        rospy.loginfo("Press [Ctrl + LeftMouseClick] to select a chessboard point. Press [Enter] on viewer to finish selection.")
        pcl_pointcloud = pcl.PointCloud(points[:,:3].astype(np.float32))
        region_growing = pcl_pointcloud.make_RegionGrowing(searchRadius=self.chessboard_diagonal/5.0)
        indices = []
        viewer.wait()
        indices = viewer.get('selected')
        if (len(indices) < 1):
            rospy.logwarn("No point selected!")
            # self.terminate_all()
            viewer.close()
            return None, None, None
        rg_indices = region_growing.get_SegmentFromPoint(indices[0])
        points_color = np.zeros(len(points))
        points_color[rg_indices] = 1
        viewer.attributes(points_color)
        # viewer.wait()
        chessboard_pointcloud = pcl_pointcloud.extract(rg_indices)
        plane_model = pcl.SampleConsensusModelPlane(chessboard_pointcloud)
        pcl_RANSAC = pcl.RandomSampleConsensus(plane_model)
        pcl_RANSAC.set_DistanceThreshold(self.args.plane_dist_thresh)
        pcl_RANSAC.computeModel()
        inliers = pcl_RANSAC.get_Inliers()
        # random select a fixed number of lidar points to reduce computation time
        if (self.args.select_lidar_points > 0):
            inliers = np.random.choice(inliers, self.args.select_lidar_points)
        chessboard_points = np.asarray(chessboard_pointcloud)
        pca = PCA()
        pca.fit(chessboard_points[inliers])
        normal_vector = pca.components_[2].reshape(1,-1)
        print("normal_vector.shape:", normal_vector.shape)
        # proc_results.put([np.asarray(chessboard_points[inliers, :3])])
        points_color = np.ones(len(chessboard_points))
        points_color[inliers] = 2
        viewer.load(chessboard_points[:,:3])
        # if 'left' in self.lidar_name[1]:
        #     viewer.set(lookat=(0,2,4))
        #     viewer.set(phi=0)
        # elif 'right' in self.lidar_name[1]:
        #     viewer.set(lookat=(0,2,4))
        #     viewer.set(phi=-3.3)
        # else:
        #     viewer.set(lookat=(0,0,4))
        #     viewer.set(phi=-1.57)
        # viewer.set(lookat=(0,0,4))
        # viewer.set(phi=-1.57)
        # viewer.set(theta=0.4)
        viewer.set(r=2)
        viewer.set(floor_level=0)
        viewer.set(point_size=0.02)
        viewer.attributes(points_color)
        rospy.loginfo("Check the chessboard segmentation in the viewer.")
        viewer.wait()
        viewer.close()
        return np.asarray(chessboard_points[inliers, :3]), normal_vector, normal_vector.dot(np.mean(chessboard_points[inliers], axis=0).reshape(-1,1))[0][0]

    '''
    Calibrate the LiDAR and image points using OpenCV PnP RANSAC
    Requires minimum 5 point correspondences

    Inputs:
        points2D - [numpy array] - (N, 2) array of image points
        points3D - [numpy array] - (N, 3) array of 3D points

    Outputs:
        Extrinsics saved in PKG_PATH/CALIB_PATH/extrinsics.npz
    '''
    def calibrate(self):
        # 1. linear calibration
        if (self.slave_to_master_R is None or self.slave_to_master_T is None):
            rospy.loginfo("Step 1: Linear calibration.")
            total_lidar_points = np.row_stack(self.slave_lidar_points)
            mean_lidar_points = total_lidar_points.mean(axis=0)
            # print("mean_lidar_points:", mean_lidar_points.shape)
            diff_lidar_points = total_lidar_points - mean_lidar_points
            scale_lidar_points = np.abs(diff_lidar_points).max(axis=0).mean()
            # print("scale_lidar_points:", scale_lidar_points)
            A = None
            b = None
            for i in range(self.frame_count):
                lidar_points = (self.slave_lidar_points[i] - mean_lidar_points) / scale_lidar_points
                # r3 = self.chessboard_to_camera_R[i][:,2].reshape(-1,1) # last column of R (normal vector of chessboard plane in camera coordinate)
                r3 = self.master_normal_vector[i]
                n_points = lidar_points.shape[0]
                if (A is None):
                    A = np.c_[r3[0]*lidar_points, r3[0]*np.ones((n_points,1)), 
                        r3[1]*lidar_points, r3[1]*np.ones((n_points,1)), 
                        r3[2]*np.ones((n_points,1)), r3[2]*lidar_points]
                    # print("A form None to", A.shape)
                    b = self.master_centroids[i] / scale_lidar_points * np.ones((n_points,1))
                else:
                    A = np.r_[A, np.c_[r3[0]*lidar_points, r3[0]*np.ones((n_points,1)), 
                        r3[1]*lidar_points, r3[1]*np.ones((n_points,1)), 
                        r3[2]*np.ones((n_points,1)), r3[2]*lidar_points]]
                    b = np.r_[b, self.master_centroids[i] / scale_lidar_points * np.ones((n_points,1))]
            print("A:", A.shape, ". b:", b.shape)
            mm = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b).squeeze()
            tR = np.array([[mm[0],mm[1],mm[2]],[mm[4],mm[5],mm[6]],[mm[9],mm[10],mm[11]]])
            uu, dd, vv = np.linalg.svd(tR)
            self.slave_to_master_R = uu.dot(vv.T)
            self.slave_to_master_T = scale_lidar_points*mm[[3,7,8]].reshape(-1,1) - self.slave_to_master_R.dot(mean_lidar_points.reshape(-1,1))
            print("R:", self.slave_to_master_R)
            print("T:", self.slave_to_master_T)
            # 2. Non-linear calibration
            rospy.loginfo("Step 2: Non-linear calibration.")

        rvec_and_tvec0 = np.squeeze(np.r_[cv2.Rodrigues(self.slave_to_master_R)[0],self.slave_to_master_T])
        print("initial value [rvec_and_tvec0]:", rvec_and_tvec0)
        optimize_ret = optimize.least_squares(self.calibProjErrLidar, rvec_and_tvec0) #, bounds=(-3.14,3.14))
        rvec_and_tvec = optimize_ret.x
        self.success = optimize_ret.success
        print("camera and lidar chessboard plane error:", np.mean(optimize_ret.fun))
        # print("original optimize_ret:", optimize_ret)
        print("optimized value [rvec_and_tvec]:", rvec_and_tvec)
        print("optimization success status:", self.success)
        self.slave_to_master_R = cv2.Rodrigues(rvec_and_tvec[0:3])[0]
        self.slave_to_master_T = rvec_and_tvec[3:6].reshape(-1,1)
        print("R:", self.slave_to_master_R)
        print("T:", self.slave_to_master_T)

        euler = euler_from_matrix(self.slave_to_master_R)
        
        # Save extrinsics
        np.savez(os.path.join(self.folder, 'extrinsics_'+self.lidar_name[1]+'_and_'+self.lidar_name[1]+'.npz'),
            euler=euler, R=self.slave_to_master_R, T=self.slave_to_master_T.T)
        # writeYAMLFile(os.path.join(self.folder, 'extrinsics_'+self.lidar_name[1]+'_and_'+self.args.lidar_name+'.yaml'), {'euler':list(euler), 'R':self.slave_to_master_R.tolist(), 'T':self.slave_to_master_T.T.tolist()})

        calibration_string = cameraLidarCalibrationYamlBuf(self.slave_to_master_R, self.slave_to_master_T, self.camera_matrix)
        if (not os.path.exists(self.folder)):
            os.mkdir(self.folder)
        with open(os.path.join(self.folder, 'extrinsics_'+self.lidar_name[1]+'_and_'+self.lidar_name[1]+'.yaml'), 'w') as f:
            f.write(calibration_string)

    def calibProjErrLidar(self, rvec_and_tvec):
        err = None
        rotation_matrix = np.squeeze(cv2.Rodrigues(rvec_and_tvec[0:3])[0])
        for i in range(self.frame_count):
            r3 = self.chessboard_to_camera_R[i][:,2].reshape(-1,1)
            n_points = len(self.slave_lidar_points[i])
            diff = r3.T.dot(rotation_matrix.dot(self.slave_lidar_points[i].T) + rvec_and_tvec[3:6].reshape(-1,1) - self.chessboard_to_camera_tvecs[i])
            if (self.use_L1_error):
                curr_err = np.abs(diff).mean()
            else:
                curr_err = np.sqrt(np.square(diff).mean())
            if (err is None):
                err = curr_err
            else:
                err = np.r_[err, curr_err]
        return err

    '''
    Projects the point cloud on to the image plane using the extrinsics

    Inputs:
        img_msg - [sensor_msgs/Image] - ROS sensor image message
        pointcloud_msg - [sensor_msgs/PointCloud2] - ROS pointcloud_msg PCL2 message
        image_pub - [sensor_msgs/Image] - ROS image publisher

    Outputs:
        Projected points published on /sensors/camera/camera_lidar topic
    '''
    def project_point_cloud(self, master_pointcloud_msg, slave_pointcloud_msg):#points3D, img, image_pub):

        if (self.slave_to_master_R is None or self.slave_to_master_T is None):
            extrinsic_fname = 'extrinsics_'+self.lidar_name[1]+'_and_'+self.lidar_name[1]+'.yaml'
            rospy.loginfo("Reading extrinsics from %s" % os.path.join(self.folder, extrinsic_fname))
            extrinsics = readYAMLFile(os.path.join(self.folder, extrinsic_fname))
            if ('RT' in extrinsics):
                self.slave_to_master_RT = np.array(extrinsics['RT']['data']).reshape(extrinsics['RT']['rows'], extrinsics['RT']['cols'])
                print("RT:", self.slave_to_master_RT.shape)
                self.slave_to_master_R = self.slave_to_master_RT[:3,:3]
                self.slave_to_master_T = self.slave_to_master_RT[:3,3].reshape(-1,1)
            else:
                self.slave_to_master_R = np.array(extrinsics['R'])
                self.slave_to_master_T = np.array(extrinsics['T'])
        # Transform the point cloud
        # try:
        #     transform = TF_BUFFER.lookup_transform('world', 'vehicle_frame', rospy.Time())
        #     pointcloud_msg = do_transform_cloud(pointcloud_msg, transform)
        # except tf2_ros.LookupException:
        #     pass

        # Extract points from message
        master_points = ros_numpy.point_cloud2.pointcloud2_to_array(master_pointcloud_msg)
        master_points = np.asarray(master_points.tolist())
        if (len(master_points.shape)>2):
            master_points = master_points.reshape(-1, master_points.shape[-1])
        master_points = master_points[~np.isnan(master_points).any(axis=1), :]
        
        slave_points = ros_numpy.point_cloud2.pointcloud2_to_array(slave_pointcloud_msg)
        slave_points = np.asarray(slave_points.tolist())
        if (len(slave_points.shape)>2):
            slave_points = slave_points.reshape(-1, slave_points.shape[-1])
        slave_points = slave_points[~np.isnan(slave_points).any(axis=1), :]

        # if 'left' in self.lidar_name[1]:
        #     inrange = np.where((points3D[:, 0] < -3) &
        #                     (points3D[:, 0] > -14) &
        #                     (points3D[:, 1] < 5) &
        #                     (points3D[:, 1] > -5) &
        #                     (points3D[:, 2] < 5) &
        #                     (points3D[:, 2] > 0.2))
        # elif 'right' in self.lidar_name[1]:
        #     inrange = np.where((points3D[:, 0] > 3) &
        #                     (points3D[:, 0] < 14) &
        #                     (points3D[:, 1] < 5) &
        #                     (points3D[:, 1] > -5) &
        #                     (points3D[:, 2] < 4) &
        #                     (points3D[:, 2] > 0.2))
        # else:
        #     # Filter points in front of camera
        #     inrange = np.where((np.abs(points3D[:, 0]) < 7) &
        #                     (points3D[:, 1] < 16) &
        #                     (points3D[:, 1] > 2.5) &
        #                     (points3D[:, 2] < 4) &
        #                     (points3D[:, 2] > -0.5))
        inrange = np.where((np.abs(master_points[:, 0]) < 7) &
                        (master_points[:, 1] < 16) &
                        (master_points[:, 1] > 2.5) &
                        (master_points[:, 2] < 4) &
                        (master_points[:, 2] > -0.5))
        # master_points = master_points[inrange[0]]
        inrange = np.where((np.abs(slave_points[:, 0]) < 7) &
                        (slave_points[:, 1] < 16) &
                        (slave_points[:, 1] > 2.5) &
                        (slave_points[:, 2] < 4) &
                        (slave_points[:, 2] > -0.5))
        # slave_points = slave_points[inrange[0]]
        max_intensity = np.max(master_points[:, -1])
        # Color map for the points
        cmap = matplotlib.cm.get_cmap('jet')
        if (max_intensity > 0):
            colors = cmap(points[:, -1] / max_intensity) * 255
        else:
            colors = cmap(points[:, -1] / 1) * 255
        points = points[:,:3]
        points_T = (self.slave_to_master_R.dot(slave_points.T) + self.slave_to_master_T.reshape(-1,1)) # (3, N)

        points_transformed = self.camera_matrix.dot(points_T).T
        inrange = np.where((points_transformed[:, 0] >= 0) &
                        (points_transformed[:, 1] >= 0) &
                        (points_transformed[:, 0] < img.shape[1]) &
                        (points_transformed[:, 1] < img.shape[0]))
        points_transformed = points_transformed[inrange[0]].round().astype('int')
        master_points += points_transformed
        viewer = pptk.viewer(master_points[:, :3])
        # if 'left' in self.lidar_name[1]:
        #     viewer.set(lookat=(0,2,4))
        #     viewer.set(phi=0)
        # elif 'right' in self.lidar_name[1]:
        #     viewer.set(lookat=(0,2,4))
        #     viewer.set(phi=-3.3)
        # else:
        #     viewer.set(lookat=(0,0,4))
        #     viewer.set(phi=-1.57)
        viewer.set(lookat=(0,0,4))
        viewer.set(phi=-1.57)
        viewer.set(theta=0.4)
        viewer.set(r=4)
        viewer.set(floor_level=0)
        viewer.set(point_size=0.02)

        # visualize
        # rospy.loginfo("lidar image published.")
        # print("\rlidar image published.", end="")
        # sys.stdout.flush()

    '''
    Callback function to publish project image and run calibration

    Inputs:
        image - [sensor_msgs/Image] - ROS sensor image message
        camera_info - [sensor_msgs/CameraInfo] - ROS sensor camera info message
        velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
        image_pub - [sensor_msgs/Image] - ROS image publisher

    Outputs: None
    '''
    def lidar_callback(self, master_pointcloud_msg, slave_pointcloud_msg):
        global FIRST_TIME, PAUSE, TF_BUFFER, TF_LISTENER, PROJECT_MODE, WATCH_LIVESTREAM_OUTCOME, REMOVE_LAST_FRAME
        # if self.callback_running:
        #     return
        # self.callback_running = True

        if FIRST_TIME:
            FIRST_TIME = False
            # TF listener
            # rospy.loginfo('Setting up static transform listener')
            # TF_BUFFER = tf2_ros.Buffer()
            # TF_LISTENER = tf2_ros.TransformListener(TF_BUFFER)
            if (not os.path.exists(os.path.join(PKG_PATH, CALIB_PATH, '_'.join(self.lidar_name)))):
                os.mkdir(os.path.join(PKG_PATH, CALIB_PATH, '_'.join(self.lidar_name)))

        elif REMOVE_LAST_FRAME:
            REMOVE_LAST_FRAME = False
            if (len(self.master_lidar_points)>0):
                self.master_lidar_points.pop()
                self.slave_lidar_points.pop()
                self.lidar_points_num = len(self.master_lidar_points)
                rospy.loginfo("Current number of frames: %d" % len(self.master_lidar_points))
            else:
                rospy.logwarn("No frame stored.")
            start_keyboard_handler()

        # Projection/display mode
        elif PROJECT_MODE or (WATCH_LIVESTREAM_OUTCOME):
            # rospy.loginfo("Entering project_point_cloud")

            self.project_point_cloud(master_pointcloud_msg, slave_pointcloud_msg)
            if self.calibrate_mode:
                PROJECT_MODE = False
            if (WATCH_LIVESTREAM_OUTCOME and PAUSE):
                PAUSE = False
                start_keyboard_handler()

        # Calibration mode
        elif PAUSE:
            # Resume listener
            with KEY_LOCK: PAUSE = False
            rospy.loginfo("Frame %d" % (self.frame_count + 1))

            self.lidar_points_num = len(self.master_lidar_points)



            # Create GUI processes
            now = rospy.get_rostime()
            # proc_results = multiprocessing.Queue()
            # img_proc = multiprocessing.Process(target=self.extract_points_2D, args=[image_msg, now, proc_results])
            # self.procs.append(img_proc)
            # proc_results.append(proc_pool.apply_async(self.extract_lidar_points, args=(pointcloud_msg, now, self.lidar_points,)))
            # pcl_proc = multiprocessing.Process(target=self.extract_lidar_points, args=[pointcloud_msg, now, proc_results])
            # self.procs.append(pcl_proc)
            # pool.close()  # 关闭进程池，不再接受新的进程
            # pool.join()  # 主进程阻塞等待子进程的退出
            # rospy.loginfo("Starting sub threads.")
            # img_proc.start()
            # pcl_proc.start()
            cur_master_lidar_points, cur_master_normal_vector, cur_master_centroid = self.extract_lidar_points(master_pointcloud_msg)
            cur_slave_lidar_points = None
            if (cur_master_lidar_points is not None):
                print("cur_master_centroid:", cur_master_centroid, cur_master_centroid.shape)
                cur_slave_lidar_points, cur_slave_normal_vector, cur_slave_centroid = self.extract_lidar_points(slave_pointcloud_msg)
            # img_proc.join()
            # pcl_proc.join()
            # rospy.loginfo("All sub threads ended.")
            # results = []
            # results_valid = True
            # if (proc_results.empty()):
            #     results_valid = False
            # else:
            #     try:
            #         for proc in self.procs:
            #             results.append(proc_results.get_nowait())
            #     except Exception:
            #         rospy.logwarn("Get results error. Pass this frame.")
            #         results_valid = False
            # if (results_valid and len(results)>0 and cur_lidar_points is not None):
            if (cur_master_lidar_points is not None and cur_slave_lidar_points is not None):
                self.master_lidar_points.append(cur_master_lidar_points)
                self.slave_lidar_points.append(cur_slave_lidar_points)
                self.master_normal_vector.append(cur_master_normal_vector)
                self.master_centroids.append(cur_master_centroid)
                self.frame_count += 1
                
            # del self.procs
            # self.procs = []

            # Calibrate for existing corresponding points
            if (self.frame_count >= self.calibrate_min_frames and (self.lidar_points_num < len(self.master_lidar_points))):
                rospy.loginfo("Entering calibration")
                self.calibrate()
            if (self.success):
                self.calibration_finished = True
                PROJECT_MODE = True

            start_keyboard_handler()

        # self.callback_running = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="enable calibration mode")
    parser.add_argument('--project', action="store_true", help="enable projection mode")
    parser.add_argument('--lidar_name', type=str, nargs='+')
    parser.add_argument('--transform_pointcloud', action="store_true")
    parser.add_argument('--pointcloud_topic', type=str, nargs='+')
    # parser.add_argument('--camera_lidar_topic', type=str, default='/camera_lidar_topic', help="topic name of image message on which lidar pointcloud is projected")
    parser.add_argument('--select_lidar_points', type=int, default='100', help="(ADVANCED) number of points choosen on chessboard plane. reduce this number could help this program run faster")
    parser.add_argument('--plane_dist_thresh', type=float, default='0.01', help="(ADVANCED) distance threshhold for RANSAC plane fit")
    parser.add_argument('--calibrate_min_frames', type=int, default='4', help="(ADVANCED) least number of frames to perform calibration")
    parser.add_argument('--chessboard_size', type=str, default='6x8', help="(number of corners in a row)x(number of corners in a column)")
    parser.add_argument('--square_size', type=float, default='0.12', help="length (in meter) of a side of square")
    parser.add_argument('--use_L1_error', action="store_true", help="(ADVANCED) use L1 error for lidar and camera projection error")
    parser.add_argument('--tracking_mode', action="store_true", help="(ADVANCED) unimplemented. track the chessboard in pointcloud")
    args = parser.parse_args()

    # Calibration mode
    if args.calibrate:
        pointcloud_topic = args.pointcloud_topic
        # camera_lidar_topic = args.camera_lidar_topic
        PROJECT_MODE = False
    # Projection mode
    else:
        pointcloud_topic = args.pointcloud_topic
        # camera_lidar_topic = args.camera_lidar_topic
        PROJECT_MODE = args.project

    # Start DoubleLiDARCalibrator
    calibrator = DoubleLiDARCalibrator(args, pointcloud_topic)#, camera_lidar_topic)
