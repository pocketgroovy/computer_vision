"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
from numpy.random.mtrand import randn


# def show_image(img, img2=None):
#     if img2 is not None:
#         cv2.imshow("img", np.hstack((img, img2)))
#     else:
#         cv2.imshow("img", img)
#     cv2.waitKey(0)


class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.])  # state
        self.covariance = np.eye(4)
        delta_t = 1
        self.Dt = np.array([[1, 0, delta_t, 0], [0, 1, 0, delta_t], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.Mt = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.process_noise = Q
        self.measurement_noise = R

    def predict(self):
        transition_mat = self.Dt
        self.state = transition_mat @ self.state
        self.covariance = (transition_mat @ self.covariance @ transition_mat.T) + self.process_noise

    def correct(self, meas_x, meas_y):
        state = self.state
        noise = self.measurement_noise
        covariance = self.covariance
        M = self.Mt
        kalman_gain = (covariance @ M.T) @ np.linalg.inv(M @ covariance @ M.T + noise)
        Y = ([meas_x, meas_y])
        self.state = state + (kalman_gain @ (Y - (M @ state)))
        I = np.eye(4)
        self.covariance = (I - kalman_gain @ M) @ covariance

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        np.random.seed(0)

        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        t_x = self.template_rect.get("x")
        t_y = self.template_rect.get("y")
        self.template = template
        t_h, t_w, _ = template.shape
        t_center = (t_w//2 + t_x, t_h//2 + t_y)
        self.frame = frame
        self.particles = self.create_gaussian_particles(mean=t_center, std=(t_w, t_h), N=self.num_particles) # Initialize your particles array. Read the docstring.
        self.weights = np.full(self.num_particles, 1/self.num_particles)  # Initialize your weights array. Read the docstring.
        # Initialize any other components you may need when designing your filter.

    def create_gaussian_particles(self, mean, std, N):
        particles = np.empty((N, 2))
        particles[:, 0] = mean[0] + (randn(N) * std[0])
        particles[:, 1] = mean[1] + (randn(N) * std[1])
        return particles.astype(np.int)

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        # ref https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy
        # mse = ((template.astype(np.float64) - frame_cutout.astype(np.float64))**2).mean() # for Grayscale
        mse = ((template.astype(np.int).sum(axis=2) - frame_cutout.astype(np.int).sum(axis=2))**2).mean() # for luma
        similarity = np.exp(-1 * (mse / (2 * (self.sigma_exp ** 2))))
        return similarity

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        particles = self.particles
        sampled_particles = particles[np.random.choice(particles.shape[0], particles.shape[0], replace=True,
                                                       p=self.weights), :]
        return sampled_particles

    def add_noise(self, particles):
        # particles = particles + np.random.randint((self.sigma_dyn**2)*-1, self.sigma_dyn**2, particles.shape)
        particles = particles + np.random.randint((self.sigma_dyn)*-1, self.sigma_dyn, particles.shape)

        return particles

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        t_width = self.template_rect.get("w")
        half_width = t_width // 2
        t_height = self.template_rect.get("h")
        half_height = t_height // 2
        template = get_luma_image(self.template.copy())
        luma_frame = get_luma_image(frame.copy())
        # template = cv2.cvtColor(self.template.copy(), cv2.COLOR_BGR2GRAY)
        # luma_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        self.particles = self.resample_particles()
        self.particles = self.add_noise(self.particles)
        x_particles = self.particles[:, 0]
        y_particles = self.particles[:, 1]
        total_weight = 0
        weights = self.weights
        for i in range(self.num_particles):
            x_start = int(x_particles[i]-half_width)
            x_end = int(x_particles[i]+half_width)
            y_start = int(y_particles[i] - half_height)
            y_end = int(y_particles[i]+half_height)
            frame_cutout = luma_frame[y_start:y_end, x_start:x_end]
            if frame_cutout.shape == template.shape:
                weights[i] = self.get_error_metric(template, frame_cutout)
            else:
                weights[i] = 0
            total_weight += weights[i]
        self.weights = weights / total_weight
        print(total_weight)

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """
        x_weighted_mean = 0
        y_weighted_mean = 0
        t_width = self.template_rect.get("w")
        half_width = t_width // 2
        t_height = self.template_rect.get("h")
        half_height = t_height // 2

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
            mark_location(frame_in, (self.particles[i, 0], self.particles[i, 1]))

        left_x = int(x_weighted_mean - half_width)
        top_y = int(y_weighted_mean - half_height)
        bottom_y = int(y_weighted_mean + half_height)
        right_x = int(x_weighted_mean + half_width)
        markers = [(left_x, top_y), (left_x, bottom_y), (right_x, top_y), (right_x, bottom_y)]
        draw_box(frame_in, markers)
        x_weighted_mean_distance = 0
        y_weighted_mean_distance = 0
        for i in range(self.num_particles):
            x_weighted_mean_distance += abs(x_weighted_mean - self.particles[i, 0])
            y_weighted_mean_distance += abs(y_weighted_mean - self.particles[i, 1])

        print("weighted_mean_distance")
        print(x_weighted_mean)
        print(y_weighted_mean)

        draw_circle(frame_in, (int(x_weighted_mean), int(y_weighted_mean)),
                    int(np.sqrt((int((x_weighted_mean_distance/self.num_particles)**2) +
                                 int((y_weighted_mean_distance/self.num_particles)**2)))))


def draw_circle(image, center, radius, thickness=1):
    color = (200, 200, 200)
    cv2.circle(image, center, radius, color, thickness)


    # from ps3
def draw_box(image, markers, thickness=1):
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


def get_luma_image(image):
    t_blue = image[:, :, 0] * 0.12
    t_green = image[:, :, 1] * 0.58
    t_red = image[:, :, 2] * 0.3
    image[:, :, 0] = t_blue
    image[:, :, 1] = t_green
    image[:, :, 2] = t_red
    return image


# from ps3
def mark_location(image, pt):
    color = (0, 50, 255)
    cv2.circle(image, pt, 3, color, -1)


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        raise NotImplementedError


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        raise NotImplementedError