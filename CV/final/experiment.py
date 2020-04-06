import numpy as np
import cv2
import os

import motion
import final as fn
from scipy import stats

VID_DIR = 'input_video'


def load_video(video_name):
    return os.path.join(VID_DIR, video_name)


# from PS3
def video_frame_generator(filename):
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None


VIDEO_NAME = "out_video.mp4"


def final_experiment():
    theta = 50
    tau = 15
    kernel_shape = cv2.MORPH_ELLIPSE
    kernel_size = (2, 2)
    fps = 25

    video = load_video('person15_walking_d1_uncomp.avi')
    video_frames = video_frame_generator(video)
    frame1 = video_frames.__next__()
    gray_frm1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    mhi = motion.MotionHistoryImage(tau, gray_frm1)

    h, w, d = frame1.shape
    video_out = mp4_video_writer(VIDEO_NAME, (w, h), fps, False)

    motion_history = []
    fr = 1
    while frame1 is not None:
        t = fr % tau
        frame2 = video_frames.__next__()
        if frame2 is None:
            break
        gray_frm2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        if fr >= tau:
            motion_history[:-1] = motion_history[1:]
            motion_history[-1] = gray_frm1
            mode_moments = stats.mode(motion_history)
            binary_image = fn.get_frame_diff_binary(gray_frm2, mode_moments.mode[0], theta)
            cleaned_image = cleanup_image(binary_image, kernel_shape, kernel_size)
            mhi.update_m_at_t(cleaned_image, t)
        # moments = fn.get_moments_binary(mhi.MT[t])
        # central_moments = fn.get_central_moments_binary(moments, mhi.MT[t])
        # hu_moments = fn.get_hu_moments(central_moments)
            normalized = normalize_and_scale(mhi.MT[t])
            video_out.write(normalized.astype(np.uint8))
        else:
            motion_history.append(gray_frm1)
        gray_frm1 = gray_frm2
        fr += 1
    video_out.release()


def cleanup_image(binary_image, kernel_shape, kernel_size):
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    opened_image = fn.morph_open(binary_image, kernel)
    return opened_image


# from ps3
def mp4_video_writer(filename, frame_size, fps=20, is_color=True):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
        is_color (bool): frame is in color or not
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size, is_color)


#from ps4
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
final_experiment()
