#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import threading
import numpy as np
import cv2
import yaml
import glob
import argparse
from checkerboard import detect_checkerboard
from utils import *


class CameraCalibrator:
    def __init__(self, args):
        assert ('x' in args.chessboard_size)
        self.chessboard_size = tuple([int(x) for x in args.chessboard_size.split('x')])
        assert (len(self.chessboard_size) == 2)
        # chessboard refine termination criteria
        self.chessboard_corner_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.chessboard_corner_coords = np.zeros((self.chessboard_size[0]*self.chessboard_size[1],3), np.float32)
        self.chessboard_corner_coords[:,:2] = np.mgrid[0:self.chessboard_size[0],0:self.chessboard_size[1]].T.reshape(-1,2) * args.square_size
        self.image_path_list = glob.glob(args.path)
        print("Found %d images" % len(self.image_path_list))
        self.calibrate(args)

    def calibrate(self, args):
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        for fname in self.image_path_list:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size,None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(self.chessboard_corner_coords)

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.chessboard_corner_criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, self.chessboard_size, corners2,ret)
                cv2.namedWindow('img', 0)
                cv2.imshow('img',img)
                key = cv2.waitKey(0)
                if (key==13 or key==32): #按空格和回车键退出
                    cv2.destroyAllWindows()
                    continue
                elif (key == ord('q')):
                    return
            else:

                print("No chessboard corners found in the current image! Press [Enter] on the image to skip this frame.")
                print("Try another chessboard corners detection method...")
                corners, score = detect_checkerboard(gray, self.chessboard_size)
                if (corners is None):
                    print("Detection failed.")
                    continue
                corners = corners.astype(np.float32)
                objpoints.append(self.chessboard_corner_coords)
                imgpoints.append(corners)
                print("detect_checkerboard return score:", score)
                cv2.drawChessboardCorners(img, self.chessboard_size, corners, True)
                cv2.namedWindow('img', 0)
                cv2.imshow('img',img)
                key = cv2.waitKey(0)
                if (key==13 or key==32): #按空格和回车键退出
                    cv2.destroyAllWindows()
                    continue
                elif (key == ord('q')):
                    return
            # if (args.step_mode):
                # start_keyboard_handler()
                # handle_keyboard()

        cv2.destroyAllWindows()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        writeYAMLFile('right2.yaml', {'image_width':gray.shape[1], \
            'image_height':gray.shape[0], 'camera_matrix':mtx, \
            'distortion_model':'plumb_bob', 'distortion_coefficients': dist})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_mode", action="store_true")
    parser.add_argument('--path', type=str, default="./")
    parser.add_argument('--chessboard_size', type=str, default='9x7')
    parser.add_argument('--square_size', type=float, default='0.12')
    args = parser.parse_args()
    calibrator = CameraCalibrator(args)
