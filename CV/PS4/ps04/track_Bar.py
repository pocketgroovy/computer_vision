from __future__ import print_function
from __future__ import division
import cv2 as cv
import argparse

import ps4
import cv2
import numpy as np

class TrackBar:
    alpha_slider_max = 100
    title_window = 'Linear Blend'

    def __init__(self, src1, src2):
        self.src1 = src1
        self.src2 = src2

    # Utility code
    def quiver(self, u, v, scale, stride, color=(0, 255, 0)):
        img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

        for y in range(0, v.shape[0], stride):

            for x in range(0, u.shape[1], stride):
                cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                           y + int(v[y, x] * scale)), color, 1)
                cv2.circle(img_out, (x + int(u[y, x] * scale),
                                     y + int(v[y, x] * scale)), 1, color, 1)
        return img_out

    def k_size_bar(self, val):
        u, v = ps4.optic_flow_lk(self.src1, self.src2, k_size=val, k_type="")
        # Flow image
        u_v = self.quiver(u, v, scale=3, stride=10)
        cv.imshow(self.title_window, u_v)

    def sigma_bar(self, val):
        u, v = ps4.optic_flow_lk(self.src1, self.src2, k_size=20, k_type="gaussian", sigma=val)
        # Flow image
        u_v = self.quiver(u, v, scale=3, stride=10)
        cv.imshow(self.title_window, u_v)


    def Gaussian_Blur(self, val):
        blurred = cv2.GaussianBlur(self.src1, ksize=val, sigmaX=1, sigmaY=1)
        # Flow image
        cv.imshow(self.title_window, blurred)

    def start_track(self):
        cv.namedWindow(self.title_window)
        trackbar_name = 'Alpha x %d' % self.alpha_slider_max
        # cv.createTrackbar(trackbar_name, self.title_window , 0, self.alpha_slider_max, self.k_size_bar)
        # cv.createTrackbar(trackbar_name, self.title_window , 0, self.alpha_slider_max, self.sigma_bar)
        cv.createTrackbar(trackbar_name, self.title_window , 80, self.alpha_slider_max, self.Gaussian_Blur)

        # Show some stuff
        # self.k_size_bar(80)
        # self.sigma_bar(1)
        self.Gaussian_Blur(80)
        # Wait until user press some key
        cv.waitKey()