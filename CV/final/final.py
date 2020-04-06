import numpy as np
import cv2


def show_image(img, img2=None):
    if img2 is not None:
        cv2.imshow("img", np.hstack((img, img2)))
    else:
        cv2.imshow("img", img)
    cv2.waitKey(0)


def get_frame_diff_binary(cur_image, past_image, theta):
    diff = cv2.absdiff(cur_image, past_image)
    binary = convert_gray_to_binary_image(diff, theta)
    return binary


def get_moments_binary(binary_image):
    row, col = np.mgrid[:binary_image.shape[0], :binary_image.shape[1]]
    moments = {'m00': np.sum(binary_image), 'm01': np.sum(row * binary_image), 'm10': np.sum(col * binary_image),
               'm11': np.sum(col * row * binary_image)}
    moments['mean_col'] = moments['m10'] / moments['m00']
    moments['mean_row'] = moments['m01'] / moments['m00']
    return moments


def get_central_moments_binary(moments, binary_image):
    row, col = np.mgrid[:binary_image.shape[0], :binary_image.shape[1]]
    central_moments = {'mu11': np.sum((col - moments['mean_col']) * (row - moments['mean_row']) * binary_image),
                       'mu02': np.sum((row - moments['mean_row']) ** 2 * binary_image),
                       'mu20': np.sum((col - moments['mean_col']) ** 2 * binary_image),
                       'mu12': np.sum((col - moments['mean_col']) * (row - moments['mean_row']) ** 2 * binary_image),
                       'mu21': np.sum((col - moments['mean_col']) ** 2 * (row - moments['mean_row']) * binary_image),
                       'mu22': np.sum((col - moments['mean_col']) ** 2 * (row - moments['mean_row']) ** 2 * binary_image),
                       'mu03': np.sum((row - moments['mean_row']) ** 3 * binary_image),
                       'mu30': np.sum((col - moments['mean_col']) ** 3 * binary_image)}
    return central_moments


def convert_gray_to_binary_image(gray_image, theta):
    thresh, bandw = cv2.threshold(gray_image, theta, 1, cv2.THRESH_BINARY)
    return bandw


def morph_open(binary_image, kernel):
    img_opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    return img_opened


def erode_image(binary_image, kernel, origin=(-1, -1)):
    img_eroded = cv2.erode(binary_image, kernel, anchor=origin, iterations=1)
    return img_eroded


def dilate_image(binary_image, kernel, origin=(-1, -1)):
    img_dilated = cv2.dilate(binary_image, kernel, anchor=origin, iterations=1)
    return img_dilated


def get_hu_moments(central_moments):
    mu11 = central_moments['mu11']
    mu02 = central_moments['mu02']
    mu20 = central_moments['mu20']
    mu12 = central_moments['mu12']
    mu21 = central_moments['mu21']
    mu03 = central_moments['mu03']
    mu30 = central_moments['mu30']
    hu_moments = {'h1': mu20 + mu02, 'h2': (mu20 - mu02) ** 2 + 4 * (mu11 ** 2),
                  'h3': (mu30 - 3 * mu12) ** 2 + (3 * mu21 - mu03) ** 2, 'h4': (mu30 + mu12) ** 2 + (mu21 + mu03) ** 2,
                  'h5': (mu30 - 3 * mu12) * (mu30 + mu12) * ((mu30 + mu12) ** 2 - 3 * (mu21 + mu03) ** 2) +
                        (3 * mu21 - mu03) * (mu21 + mu03) * (3 * (mu30 + mu12) ** 2 - (mu21 + mu03) ** 2),
                  'h6': (mu20 - mu02) * ((mu30 + mu12) ** 2 - (mu21 + mu03) ** 2) + 4 * mu11 * (mu30 + mu12) * (
                          mu21 + mu03),
                  'h7': (3 * mu21 - mu03) * (mu30 + mu12) * ((mu30 + mu12) ** 2 - 3 * (mu21 + mu03) ** 2) -
                        (mu30 - 3 * mu12) * (mu21 + mu03) * (3 * (mu30 + mu12) ** 2 - (mu21 + mu03) ** 2)}
    return hu_moments


def get_normalized_scale_invariant_moments(central_moments, moments):
    nu_moments = {'nu11': central_moments['mu11'] / moments['m00'] ** (2 / 2 + 1),
                  'nu12': central_moments['mu12'] / moments['m00'] ** (3 / 2 + 1),
                  'nu21': central_moments['mu21'] / moments['m00'] ** (3 / 2 + 1),
                  'nu20': central_moments['mu20'] / moments['m00'] ** (2 / 2 + 1),
                  'nu03': central_moments['mu03'] / moments['m00'] ** (3 / 2 + 1),
                  'nu30': central_moments['mu30'] / moments['m00'] ** (3 / 2 + 1),
                  'nu02': central_moments['mu02'] / moments['m00'] ** (2 / 2 + 1),
                  'nu22': central_moments['mu22'] / moments['m00'] ** (4 / 2 + 1)}
    return nu_moments
