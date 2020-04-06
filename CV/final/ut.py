import unittest
import final as fn
import numpy as np
import cv2
import os
import pickle
import motion

input_dir = "test_images"


class Final(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_image1 = cv2.imread(os.path.join(input_dir, "dot1.png"), 0)
        cls.test_image2 = cv2.imread(os.path.join(input_dir, "dot2.png"), 0)
        cls.binary_array = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1]])

    def test_get_moments(self):
        moments = fn.get_moments_binary(self.binary_array)
        self.assertTrue(moments['m00'] == 6)
        self.assertTrue(moments['m10'] == 6)
        self.assertTrue(moments['m01'] == 9)

    def test_get_central_moment(self):
        moments = fn.get_moments_binary(self.binary_array)
        central_moments = fn.get_central_moments_binary(moments, self.binary_array)
        self.assertTrue(central_moments['mu11'] == 1.0)
        self.assertTrue(central_moments['mu02'] == 5.5)
        self.assertTrue(central_moments['mu20'] == 4.0)
        self.assertTrue(central_moments['mu12'] == 2.0)
        self.assertTrue(central_moments['mu21'] == 1.0)
        self.assertTrue(central_moments['mu03'] == 0.0)
        self.assertTrue(central_moments['mu30'] == 0.0)

    def test_get_frame_diff_binary(self):
        dot_gray1 = self.test_image1
        dot_gray2 = self.test_image2
        theta = 127
        binary_image = fn.get_frame_diff_binary(dot_gray1, dot_gray2, theta)
        self.assertTrue(binary_image.min() == 0)
        self.assertTrue(binary_image.max() == 1)

    def test_erode_image(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        binary_before = np.array([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [1, 1, 1, 1, 1]])
        eroded = fn.erode_image(binary_before.astype(np.uint8), kernel)
        expected = np.array([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0]])
        self.assertTrue((eroded == expected).all())

    def test_dilate_image(self):
        kernel = np.array([[1, 0], [1, 1]]).astype(np.uint8)
        binary_before = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        dilated = fn.dilate_image(binary_before.astype(np.uint8), kernel)
        expected = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
        self.assertTrue((dilated == expected).all())

    def test_morph_open(self):
        dot_gray1 = self.test_image1
        dot_gray2 = self.test_image2
        theta = 127
        binary_image = fn.get_frame_diff_binary(dot_gray1, dot_gray2, theta)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened_image = fn.morph_open(binary_image, kernel)
        expected = load("opened")
        self.assertTrue((opened_image == expected).all())

    def test_init_MHI(self):
        frame = np.zeros((5, 6), np.uint8)
        tau = 15
        mhi = motion.MotionHistoryImage(tau, frame)
        expected = np.zeros((15, 5, 6))
        self.assertTrue((mhi.MT == expected).all())

    def test_mhi_update_m_at_t(self):
        frame = self.binary_array
        tau = 15
        mhi = motion.MotionHistoryImage(tau, frame)
        t = 1
        mhi.update_m_at_t(self.binary_array, t)
        expected = np.array([[0.0, 15.0, 0.0], [15.0, 0.0, 15.0], [15.0, 15.0, 0.0], [0.0, 0.0, 15.0]])
        self.assertTrue((mhi.MT[t, :, :] == expected).all())
        next_frame = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1]])
        t = 2
        mhi.update_m_at_t(next_frame, t)
        expected = np.array([[0.0, 14.0, 15.0], [14.0, 15.0, 15.0], [14.0, 15.0, 0.0], [15.0, 15.0, 15.0]])
        self.assertTrue((mhi.MT[t, :, :] == expected).all())

    def test_init_MEI(self):
        frame = np.zeros((5, 6), np.uint8)
        mei = motion.MotionEnergyImage(frame)
        expected = np.zeros((5, 6))
        self.assertTrue((mei.M == expected).all())

    def test_mei_update_m(self):
        frame = self.binary_array
        mei = motion.MotionEnergyImage(frame)
        mei.update_m(self.binary_array)
        expected = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertTrue((mei.M == expected).all())
        next_frame = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1]])
        mei.update_m(next_frame)
        expected = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        self.assertTrue((mei.M == expected).all())

    def test_get_humoments(self):
        moments = fn.get_moments_binary(self.binary_array)
        central_moments = fn.get_central_moments_binary(moments, self.binary_array)
        hu_moments = fn.get_hu_moments(central_moments)
        self.assertTrue(hu_moments['h1'], 9.5)
        self.assertTrue(hu_moments['h2'], 6.25)
        self.assertTrue(hu_moments['h3'], 45)
        self.assertTrue(hu_moments['h4'], 5.0)
        self.assertTrue(hu_moments['h5'], 21.0)
        self.assertTrue(hu_moments['h6'], 3.5)
        self.assertTrue(hu_moments['h7'], 72.0)


def save(obj, name):
    with open(name, 'wb') as obj_file:
        pickle.dump(obj, obj_file)


def load(name):
    with open(name, 'rb') as obj_file:
        py_obj = pickle.load(obj_file)
        return py_obj
