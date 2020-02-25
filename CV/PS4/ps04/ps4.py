"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2

def show_image(img, img2=None):
    if img2 is not None:
        cv2.imshow("img", np.hstack((img, img2)))
    else:
        cv2.imshow("img", img)
    cv2.waitKey(0)


def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """

    return cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=3, scale=1.0/8.0, borderType=cv2.BORDER_DEFAULT)


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """

    return cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=3, scale=1.0/8.0, borderType=cv2.BORDER_DEFAULT)

## try to solve calculation (A_t*A)^-1  on both sides to solve for [[u],[v]]
# def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
#     """Computes optic flow using the Lucas-Kanade method.
#
#     For efficiency, you should apply a convolution-based method.
#
#     Note: Implement this method using the instructions in the lectures
#     and the documentation.
#
#     You are not allowed to use any OpenCV functions that are related
#     to Optic Flow.
#
#     Args:
#         img_a (numpy.array): grayscale floating-point image with
#                              values in [0.0, 1.0].
#         img_b (numpy.array): grayscale floating-point image with
#                              values in [0.0, 1.0].
#         k_size (int): size of averaging kernel to use for weighted
#                       averages. Here we assume the kernel window is a
#                       square so you will use the same value for both
#                       width and height.
#         k_type (str): type of kernel to use for weighted averaging,
#                       'uniform' or 'gaussian'. By uniform we mean a
#                       kernel with the only ones divided by k_size**2.
#                       To implement a Gaussian kernel use
#                       cv2.getGaussianKernel. The autograder will use
#                       'uniform'.
#         sigma (float): sigma value if gaussian is chosen. Default
#                        value set to 1 because the autograder does not
#                        use this parameter.
#
#     Returns:
#         tuple: 2-element tuple containing:
#             U (numpy.array): raw displacement (in pixels) along
#                              X-axis, same size as the input images,
#                              floating-point type.
#             V (numpy.array): raw displacement (in pixels) along
#                              Y-axis, same size and type as U.
#     """
#     # find gradients
#     Ix = gradient_x(img_a)
#     Iy = gradient_y(img_b)
#     It = img_b - img_a
#     if k_type == "gaussian":
#         a = cv2.getGaussianKernel(ksize=k_size, sigma=sigma)
#         # Apply the above Gaussian kernel. Here
#         # the same kernel for both X and Y
#         Ix2 = cv2.sepFilter2D(Ix * Ix, -1, a, a)
#         Ixy = cv2.sepFilter2D(Ix * Iy, -1, a, a)
#         Iy2 = cv2.sepFilter2D(Iy * Iy, -1, a, a)
#         Ixt = cv2.sepFilter2D(Ix * It, -1, a, a)
#         Iyt = cv2.sepFilter2D(Iy * It, -1, a, a)
#     else:
#         filter_k = (k_size, k_size)
#         Ix2 = cv2.boxFilter(Ix * Ix, cv2.CV_64F, ksize=filter_k)
#         Ixy = cv2.boxFilter(Ix*Iy, cv2.CV_64F, ksize=filter_k)
#         Iy2 = cv2.boxFilter(Iy*Iy, cv2.CV_64F, ksize=filter_k)
#         Ixt = cv2.boxFilter(Ix*It, cv2.CV_64F, ksize=filter_k)
#         Iyt = cv2.boxFilter(Iy*It, cv2.CV_64F, ksize=filter_k)
#
#     A = np.array([[Ix2, Ixy], [Ixy, Iy2]])
#     AT = np.array([[Ix2.T, Ixy.T], [Ixy.T, Iy2.T]])
#     ATT = np.dot(AT, A)
#     # find determinant for each pixel
#     det_A = np.linalg.det(ATT).T
#     u = np.where(det_A > 1e-15, ((Ixy*Iyt) - (Iy2*Ixt))/det_A, 0)
#     v = np.where(det_A > 1e-15, ((Ixy*Ixt) - (Ix2*Iyt))/det_A, 0)
#     return u, v

def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    # find gradients
    # gaus_k =0
    # blur_sigma = 24
    # img_a = cv2.GaussianBlur(img_a, (gaus_k, gaus_k), blur_sigma)
    # img_b = cv2.GaussianBlur(img_b, (3, 3), blur_sigma)
    Ix = gradient_x(img_a)
    Iy = gradient_y(img_b)
    It = img_b - img_a
    if k_type == "gaussian":
        a = cv2.getGaussianKernel(ksize=k_size, sigma=sigma)
        # Apply the above Gaussian kernel. Here
        # the same kernel for both X and Y
        Ix2 = cv2.sepFilter2D(Ix * Ix, -1, a, a)
        Ixy = cv2.sepFilter2D(Ix * Iy, -1, a, a)
        Iy2 = cv2.sepFilter2D(Iy * Iy, -1, a, a)
        Ixt = cv2.sepFilter2D(Ix * It, -1, a, a)
        Iyt = cv2.sepFilter2D(Iy * It, -1, a, a)
    else:
        filter_k = (k_size, k_size)
        Ix2 = cv2.boxFilter(Ix*Ix, -1, ksize=filter_k)
        Ixy = cv2.boxFilter(Ix*Iy, -1, ksize=filter_k)
        Iy2 = cv2.boxFilter(Iy*Iy, -1, ksize=filter_k)
        Ixt = cv2.boxFilter(Ix*It, -1, ksize=filter_k)
        Iyt = cv2.boxFilter(Iy*It, -1, ksize=filter_k)

    A = np.array([[Ix2, Ixy], [Ixy, Iy2]]).T

    # find determinant for each pixel
    det_A = np.linalg.det(A).T
    # # # I_y2 * I_xt - I_xy *  I_yt / det
    # u = np.where(det_A > 1e-15, ((Iy2*-Ixt) + (-Ixy*-Iyt))/det_A, 0)
    # # I_x2 * I_yt - I_xy * I_xt / det
    # v = np.where(det_A > 1e-15, ((-Ixy*-Ixt) + (-Ix2*-Iyt))/det_A, 0)
    u = np.where(det_A > 1e-5, ((Ixy*Iyt) - (Iy2*Ixt))/det_A, 0)
    v = np.where(det_A > 1e-5, ((Ixy*Ixt) - (Ix2*Iyt))/det_A, 0)
    # v = np.where(det_A > 3.5454329e-5, ((Ixy*Ixt) - (Ix2*Iyt))/det_A, 0)
    return u, v


    # # # ref: https://stackoverflow.com/questions/14321092/lucas-kanade-python-numpy-implementation-uses-enormous-amount-of-memory
    # params = np.zeros(img_a.shape + (5,))  # Ix2, Iy2, Ixy, Ixt, Iyt
    # params[..., 0] = Ix * Ix  # I_x2
    # params[..., 1] = Iy * Iy  # I_y2
    # params[..., 2] = Ix * Iy  # I_xy
    # params[..., 3] = Ix * It  # I_xt
    # params[..., 4] = Iy * It  # I_yt
    # # del Ix, Iy, It
    # cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)# <-- boxFilter?
    # # del params
    # # ↓ 2 x 2 mat
    # win = 5
    # win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
    #               cum_params[2 * win + 1:, :-1 - 2 * win] -
    #               cum_params[:-1 - 2 * win, 2 * win + 1:] +
    #               cum_params[:-1 - 2 * win, :-1 - 2 * win])
    # # del cum_params
    # op_flow = np.zeros(img_a.shape + (2,))
    #
    # # ↓ get determinant
    # # I_x2 * I_y2 - I_xy^2
    # det = win_params[..., 0] * win_params[..., 1] - win_params[..., 2] ** 2
    # # ↓ u and v
    # # I_y2 * I_xt - I_xy *  I_yt / det
    # op_flow_x = np.where(det != 0,
    #                      (win_params[..., 1] * win_params[..., 3] -
    #                       win_params[..., 2] * win_params[..., 4]) / det,
    #                      0)
    # # I_x2 * I_yt - I_xy * I_xt / det
    # op_flow_y = np.where(det != 0,
    #                      (win_params[..., 0] * win_params[..., 4] -
    #                       win_params[..., 2] * win_params[..., 3]) / det,
    #                      0)
    #
    # print()


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    kernel = np.array([1/16, 4/16, 6/16, 4/16, 1/16])
    gaus_image = cv2.sepFilter2D(image, -1, kernel, kernel)
    reduced = np.array(gaus_image)[::2, ::2]
    return reduced


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    gaus_list = []
    reduced_image = image
    gaus_list.append(reduced_image)
    for level in range(1, levels):
        reduced_image = reduce_image(reduced_image)
        gaus_list.append(reduced_image)
    return gaus_list

# ref https://gist.github.com/uchidama/41d1c0a068f1d36dec2706715a7f17aa
def concat_images(im1, im2):
    dst = np.zeros(shape=(max(im1.shape[0], im2.shape[0]), im1.shape[1] + im2.shape[1]))
    dst[0:im1.shape[0], 0:im1.shape[1]] = im1
    dst[0:im2.shape[0], im1.shape[1]:im1.shape[1]+im2.shape[1]] = im2
    return dst


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    combined_image = img_list[0]
    combined_image = normalize_and_scale(combined_image)
    for image in img_list[1:]:
        norm_scaled_image = normalize_and_scale(image)
        combined_image = concat_images(combined_image, norm_scaled_image)
    return combined_image


# ref https://stackoverflow.com/questions/27125959/numpy-array-insert-alternate-rows-of-zeros
def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    expand = np.zeros([image.shape[0]*2, image.shape[1]*2])
    expand[::2, ::2] = image
    kernel = np.array([1 / 8, 4 / 8, 6 / 8, 4 / 8, 1 / 8])
    gaus_image = cv2.sepFilter2D(expand, -1, kernel, kernel)
    return gaus_image


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    laplacian_list = []
    last_image = g_pyr[-1]
    laplacian_list.append(last_image)
    expanded_image = expand_image(last_image)
    reversed_image_list = reversed(g_pyr[0:-1])
    for image in reversed_image_list:
        if image.shape[0] == expanded_image.shape[0]-1:
            expanded_image = expanded_image[:-1, :]
        if image.shape[1] == expanded_image.shape[1]-1:
            expanded_image = expanded_image[:, :-1]
        lap_im = image - expanded_image
        laplacian_list.append(lap_im)
        expanded_image = expand_image(image)
    return list(reversed(laplacian_list))


# ref https://stackoverflow.com/questions/44041157/how-to-warp-the-later-frame-to-previous-frame-using-optical-flow-image
def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    h, w = U.shape[:2]
    u_flow = U
    v_flow = V
    u_flow += np.arange(w)
    v_flow += np.arange(h)[:, np.newaxis]
    warped = cv2.remap(src=image, map1=u_flow.astype(np.float32), map2=v_flow.astype(np.float32), interpolation=interpolation, borderMode=border_mode)
    return warped


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    img_a_pyramid = gaussian_pyramid(img_a, levels)
    img_b_pyramid = gaussian_pyramid(img_b, levels)
    # get optic flow from LK stars from lowest level
    curr_img_b = None
    u = None
    v = None
    prev_u = None
    prev_v = None
    for level in range(levels-1, -1, -1):
        if u is not None and v is not None:
            double_res_u = u*2
            double_res_v = v*2
            u_expanded = expand_image(double_res_u)
            v_expanded = expand_image(double_res_v)
            prev_u = u_expanded.copy()
            prev_v = v_expanded.copy()
            next_img = img_b_pyramid[level]
        if curr_img_b is None:
            curr_img_b = img_b_pyramid[level]
        else:
            curr_img_b = warp(next_img, u_expanded, v_expanded, interpolation=interpolation, border_mode=border_mode)
        curr_img_a = img_a_pyramid[level]
        u, v = optic_flow_lk(curr_img_a, curr_img_b, k_size=k_size, k_type=k_type, sigma=sigma)
        if prev_u is not None and prev_v is not None:
            u = u + prev_u
            v = v + prev_v
    return u, v

# import numpy as np
# from scipy import signal
#
#
# def optical_flow(I1g, I2g, window_size, tau=1e-2):
#     kernel_x = np.array([[-1., 1.], [-1., 1.]])
#     kernel_y = np.array([[-1., -1.], [1., 1.]])
#     kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
#     w = window_size / 2  # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
#     I1g = I1g / 255.  # normalize pixels
#     I2g = I2g / 255.  # normalize pixels
#     # Implement Lucas Kanade
#     # for each point, calculate I_x, I_y, I_t
#     mode = 'same'
#     fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
#     fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
#     ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) +
#     signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
#
#
# u = np.zeros(I1g.shape)
# v = np.zeros(I1g.shape)
# # within window window_size * window_size
# for i in range(w, I1g.shape[0] - w):
#     for j in range(w, I1g.shape[1] - w):
#         Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
#         Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
#         It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
#         b = np.reshape(It, (It.shape[0],1)) # get b here
#         A = np.vstack((Ix, Iy)).T # get A here
#
#         if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
#         |- nu = np.matmul(np.linalg.pinv(A), b) # get velocity here
#         |- u[i,j]=nu[0]
#         |- v[i,j]=nu[1]
# return (u, v)


