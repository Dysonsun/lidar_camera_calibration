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
            calibration = yaml.load(stream)
        except yaml.YAMLError as exc:
            rospy.logerr(exc)
            sys.exit(1)

    return calibration


if __name__ == '__main__':

    # Get parameters when starting node from a launch file.
    if len(sys.argv) < 1:
        BAG_FILE = rospy.get_param('filename')
        CALIB_FILE = rospy.get_param('calib_data')
        CAMERA_INFO = rospy.get_param('camera_info')

    # Get parameters as arguments
    else:
        BAG_FILE = sys.argv[1]
        CALIB_FILE = sys.argv[2]
        CAMERA_INFO = '/rgb_left/camera_info'
        image_color = '/rgb_left/image_raw'

    # Load ROSBAG file
    rospy.loginfo('Bag Filename: %s', BAG_FILE)
    bag = rosbag.Bag(BAG_FILE, 'r')

    # # Output file
    folder = os.path.dirname(BAG_FILE)
    output_name = os.path.splitext(os.path.basename(BAG_FILE))[0] + '_updated.bag'
    OUTPUT_FILE = os.path.join(folder, output_name)
    os.mknod(OUTPUT_FILE)
    output = rosbag.Bag(OUTPUT_FILE, 'w')

    # Load calibration data
    calibration = load_calibration_data(CALIB_FILE)

    # Update calibration data
    rospy.loginfo('Updating %s data...' % CAMERA_INFO)
    for topic, msg, t in bag.read_messages():
        if topic == image_color:
            camera_info_msg = CameraInfo()
            camera_info_msg.header = msg.header
            camera_info_msg.height = msg.height
            camera_info_msg.width = msg.width
            camera_info_msg.distortion_model = "plumb_bob"
            camera_info_msg.D = calibration['distortion_coefficients']['data']
            camera_info_msg.K = calibration['camera_matrix']['data']
            camera_info_msg.R = calibration['rectification_matrix']['data']
            camera_info_msg.P = calibration['projection_matrix']['data']
            # output.write(CAMERA_INFO, camera_info_msg, msg.header.stamp if msg._has_header else t)
            output.write(CAMERA_INFO, camera_info_msg, t)
        output.write(topic, msg, t)
    rospy.loginfo('Done')

    # Close bag file
    bag.close()
    output.close()
