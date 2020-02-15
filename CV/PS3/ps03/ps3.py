"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
import math

BLUR_S = "small_blur"
BLUR_L = "large_blur"
SHARPEN = "sharpen"
LAPLACIAN = "laplacian"
SOBEL_X = "sobel_x"
SOBEL_Y = "sobel_y"


def show_image(img, img2=None):
    if img2 is not None:
        cv2.imshow("img", np.hstack((img, img2)))
    else:
        cv2.imshow("img", img)
    cv2.waitKey(0)


#ref: https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
def get_kernel(kernel):
    # construct average blurring kernels used to smooth an image
    small_blur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
    large_blur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
    sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
    laplacian = np.array((
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]), dtype="int")
    # construct the Sobel x-axis kernel
    sobel_x = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="int")
    # construct the Sobel y-axis kernel
    sobel_y = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype="int")
    kernels = [small_blur, large_blur, sharpen, laplacian, sobel_x, sobel_y]
    kernel_names = [BLUR_S, BLUR_L, SHARPEN, LAPLACIAN, SOBEL_X, SOBEL_Y]
    kernel_bank = {}
    i = 0
    for name in kernel_names:
        kernel_bank[name] = kernels[i]
        i+=1
    return kernel_bank.get(kernel)



def convolve(gray_img, kernel):
    output = cv2.filter2D(gray_img, -1, get_kernel(kernel))
    return output


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """

    raise NotImplementedError


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    height = image.shape[0]-1
    width = image.shape[1]-1
    top_left = (0, 0)
    bottom_left = (0, height)
    top_right = (width, 0)
    bottom_right = (width, height)
    return [top_left, bottom_left, top_right, bottom_right]


def apply_kmeans(vector, n_clusters=4, max_iter=10, epsilon=1.0, attempts=10):
    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    ret, label, center = cv2.kmeans(vector, n_clusters, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    return ret, label, center


def create_numbered_dict(points):
    numbered_dict = {}
    for i in range(points.size):
        numbered_dict[i] = points[i]
    return numbered_dict


def get_edge_length(a, b):
    if a[0] == b[0]:
        edge = b[0] - a[0]
    else:
        edge = math.sqrt(pow((b[0] - a[0]), 2) + pow((b[1] - a[1]), 2))
    return edge


def find_corner_assignment_in_square(x_points, y_points):
    x_dict = create_numbered_dict(x_points)
    y_dict = create_numbered_dict(y_points)
    x_sorted_dict = {k: v for k, v in sorted(x_dict.items(), key=lambda item: item[1])}

    list_keys = [k for k in x_sorted_dict]
    p1 = (x_dict[list_keys[0]], y_dict[list_keys[0]])
    p2 = (x_dict[list_keys[1]], y_dict[list_keys[1]])
    p3 = (x_dict[list_keys[2]], y_dict[list_keys[2]])
    p4 = (x_dict[list_keys[3]], y_dict[list_keys[3]])

    top_left = p1 if p1[1] <= p2[1] else p2
    bottom_left = p2 if top_left is not p2 else p1
    top_right = p3 if p3[1] <= p4[1] else p4
    bottom_right = p4 if top_right is not p4 else p3

    return [top_left, bottom_left, top_right, bottom_right]


# ref : https://stackoverflow.com/questions/50984205/how-to-find-corners-points-of-a-shape-in-an-image-in-opencv
## OK till part 4
# def find_markers(image, template=None):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     corners = cv2.GaussianBlur(gray, (31,31), 11)
#
#     corners = cv2.cornerHarris(corners, blockSize=24, ksize=15, k=0.01)
#
#     ret, dst = cv2.threshold(corners, 0.1 * corners.max(), 255, 0)
#
#     dst = np.uint8(dst)
#     ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
#
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1) #(80, 0.001) can be used also
#
#     corners = cv2.cornerSubPix(gray, np.float32(centroids), (1, 1), (-1, -1), criteria)
#
#     if len(corners) > 4:
#         # Set flags (Just to avoid line break in the code)
#         flags = cv2.KMEANS_RANDOM_CENTERS
#         # Apply KMeans
#         compactness, labels, centers = cv2.kmeans(corners[1:, :], 4, None, criteria, 10, flags)
#         x_corners = centers[:, 0]
#         y_corners = centers[:, 1]
#     elif len(corners) < 4:
#         raise RuntimeError("less 4 corners found")
#     else:
#         x_corners = corners[:, 0]
#         y_corners = corners[:, 1]
#     return find_corner_assignment_in_square(x_corners, y_corners)

# passed video frame generator test on GS
# def find_markers(image, template=None):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     corners = cv2.GaussianBlur(gray, (31,31), 1)
#
#     corners = cv2.cornerHarris(corners, blockSize=3, ksize=13, k=0.001)
#
#     ret, dst = cv2.threshold(corners, 0.1 * corners.max(), 255, 0)
#
#     dst = np.uint8(dst)
#     ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
#
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.1) #(80, 0.001) can be used also
#
#     corners = cv2.cornerSubPix(gray, np.float32(centroids), (1, 1), (-1, -1), criteria)
#
#     if len(corners) > 4:
#         # Set flags (Just to avoid line break in the code)
#         flags = cv2.KMEANS_RANDOM_CENTERS
#         # Apply KMeans
#         compactness, labels, centers = cv2.kmeans(corners[1:, :], 4, None, criteria, 10, flags)
#         x_corners = centers[:, 0]
#         y_corners = centers[:, 1]
#     elif len(corners) < 4:
#         raise RuntimeError("less 4 corners found")
#     else:
#         x_corners = corners[:, 0]
#         y_corners = corners[:, 1]
#     return find_corner_assignment_in_square(x_corners, y_corners)

# passed video frame generator test on GS
# only missing Test for find_markers with noise: circles + gaussian
# def find_markers(image, template=None):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     corners = cv2.GaussianBlur(gray, (31,31), 1) #3
#
#     corners = cv2.cornerHarris(corners, blockSize=3, ksize=13, k=0.001)
#
#     ret, dst = cv2.threshold(corners, 0.19 * corners.max(), 255, 0)#0.1
#
#     dst = np.uint8(dst)
#     ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
#
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001) #(80, 0.001) can be used also
#
#     corners = cv2.cornerSubPix(gray, np.float32(centroids), (1, 1), (-1, -1), criteria)
#
#     if len(corners) > 4:
#         # Set flags (Just to avoid line break in the code)
#         flags = cv2.KMEANS_RANDOM_CENTERS
#         # Apply KMeans
#         compactness, labels, centers = cv2.kmeans(corners[1:, :], 4, None, criteria, 10, flags)
#         x_corners = centers[:, 0]
#         y_corners = centers[:, 1]
#     elif len(corners) < 4:
#         raise RuntimeError("less 4 corners found")
#     else:
#         x_corners = corners[:, 0]
#         y_corners = corners[:, 1]
#     return find_corner_assignment_in_square(x_corners, y_corners)

# ref : https://stackoverflow.com/questions/50984205/how-to-find-corners-points-of-a-shape-in-an-image-in-opencv
def find_markers(image, template=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = cv2.GaussianBlur(gray, (31,31), 1) #3

    corners = cv2.cornerHarris(corners, blockSize=21, ksize=15, k=0.1)

    ret, dst = cv2.threshold(corners, 0.19 * corners.max(), 255, 0)#0.1

    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1) #(80, 0.001) can be used also

    corners = cv2.cornerSubPix(gray, np.float32(centroids), (1, 1), (-1, -1), criteria)

    if len(corners) > 4:
        # Set flags (Just to avoid line break in the code)
        flags = cv2.KMEANS_RANDOM_CENTERS
        # Apply KMeans
        compactness, labels, centers = cv2.kmeans(corners[1:, :], 4, None, criteria, 10, flags)
        x_corners = centers[:, 0]
        y_corners = centers[:, 1]
    elif len(corners) < 4:
        raise RuntimeError("less 4 corners found")
    else:
        x_corners = corners[:, 0]
        y_corners = corners[:, 1]
    return find_corner_assignment_in_square(x_corners, y_corners)


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    color = (255, 255, 255)

    top_left = markers[0]
    bottom_left = markers[1]
    top_right = markers[2]
    bottom_right = markers[3]
    image = cv2.line(image, top_left, bottom_left, color, thickness)
    image = cv2.line(image, bottom_left, bottom_right, color, thickness)
    image = cv2.line(image, bottom_right, top_right, color, thickness)
    image = cv2.line(image, top_right, top_left, color, thickness)
    return image


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    # inverse of homography
    inv_H = np.linalg.inv(homography)

    # #ref https://stackoverflow.com/questions/46520123/how-do-i-use-opencvs-remap-function/46524544#46524544
    h, w = imageB.shape[:2]

    # create matrix of [Xd, Yd, 1] when Xd = 0, Yd = 0~width
    indy, indx = np.indices((h, w), dtype=np.float32)
    a_mat = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])

    # warp the coordinates of src to those of true_dst
    map_ind = inv_H.dot(a_mat)
    map_x, map_y = map_ind[:-1] / map_ind[-1]
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)
    alpha = 0.5
    beta = 1
    dst = cv2.remap(imageA, map_x, map_y, cv2.INTER_LINEAR)
    blended = cv2.addWeighted(imageB, alpha, dst, beta, 0)
    return blended


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    dst_top_left = dst_points[0]
    dst_bottom_left = dst_points[1]
    dst_top_right = dst_points[2]
    dst_bottom_right = dst_points[3]

    H = np.zeros((3, 3))
    A = np.zeros((8, 8))
    b = np.asarray([dst_top_left[0], dst_top_left[1], dst_bottom_left[0], dst_bottom_left[1],
                    dst_top_right[0], dst_top_right[1], dst_bottom_right[0], dst_bottom_right[1]])

    idx = 0
    for pt in range(0, len(dst_points)):
        src = src_points[pt]
        dst = dst_points[pt]
        A[idx, :] = [src[0], src[1], 1,
                     0, 0, 0,
                     (-1*src[0]*dst[0]), (-1*src[1]*dst[0])]
        A[idx+1, :] = [0, 0, 0,
                       src[0], src[1], 1,
                       (-1*src[0]*dst[1]), (-1*src[1]*dst[1])]
        idx += 2
    h = np.linalg.solve(A, b)
    H[0, 0] = h[0]
    H[0, 1] = h[1]
    H[0, 2] = h[2]
    H[1, 0] = h[3]
    H[1, 1] = h[4]
    H[1, 2] = h[5]
    H[2, 0] = h[6]
    H[2, 1] = h[7]
    H[2, 2] = 1
    return H


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None
