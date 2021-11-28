#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.1
Date    : Jan 18, 2019

Description:
Script to update the camera calibration data into the ROSBAG file
Ensure that this file has executable permissions

Example Usage:
$ rosrun lidar_camera_calibration update_camera_info.py rosbag.bag calibration.yaml

Notes:
Make sure this file has executable permissions:
$ chmod +x update_camera_info.py
'''

# Python 2/3 compatibility
from __future__ import print_function

# Built-in modules
import os
import sys
import yaml
import numpy as np
# ROS modules
PKG = 'lidar_camera_calibration'
import roslib; roslib.load_manifest(PKG)
import rosbag
import rospy
from sensor_msgs.msg import CameraInfo


def load_calibration_data(filename):
    # Open calibration file
    with open(filename, 'r') as stream:
        try:
            calibration = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            rospy.logerr(exc)
            sys.exit(1)

    return calibration


if __name__ == '__main__':

    # Get parameters when starting node from a launch file.
    if len(sys.argv) < 1:
        CALIB_FILE = rospy.get_param('calib_data')
        CAMERA_INFO = rospy.get_param('camera_info')

    # Get parameters as arguments
    else:
        CALIB_FILE = sys.argv[1]
        CAMERA_INFO = '/rgb_left/camera_info'

    rospy.init_node("pub_camera_info")
    # Load calibration data
    calibration = load_calibration_data(CALIB_FILE)

    # Update calibration data
    rospy.loginfo('Publishing %s data...' % CAMERA_INFO)
    count = 0
    camera_info_msg = CameraInfo()
    # camera_info_msg.header = msg.header
    camera_info_msg.header.seq = count
    camera_info_msg.header.frame_id = '/rgb_left'
    camera_info_msg.header.stamp = rospy.Time.now()
    camera_info_msg.height = 960 # msg.height
    camera_info_msg.width = 1280 # msg.width
    camera_info_msg.distortion_model = "plumb_bob"
    camera_info_msg.D = calibration['distortion_coefficients']['data']
    camera_info_msg.K = calibration['camera_matrix']['data']
    if ('rectification_matrix' in calibration):
        camera_info_msg.R = calibration['rectification_matrix']['data']
    if ('projection_matrix' in calibration):
        camera_info_msg.P = calibration['projection_matrix']['data']
    else:
        K = np.array(camera_info_msg.K)
        # print("K:", K.shape)
        P = np.c_[K.reshape(3,3),np.zeros((3,1))]
        # print("P:", P)
        camera_info_msg.P = P.ravel().tolist()
    print(camera_info_msg)
    camera_info_pub = rospy.Publisher("/rgb_left/camera_info", CameraInfo, queue_size=1)
    rate = rospy.Rate(1)
    while(not rospy.is_shutdown()):
        rate.sleep()
        camera_info_msg.header.seq = count
        camera_info_msg.header.frame_id = '/rgb_left'
        camera_info_msg.header.stamp = rospy.Time.now()
        camera_info_pub.publish(camera_info_msg)
        count += 1
