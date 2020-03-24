"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier


def show_image(img, img2=None):
    if img2 is not None:
        cv2.imshow("img", np.hstack((img, img2)))
    else:
        cv2.imshow("img", img.astype("uint8"))
    cv2.waitKey(0)


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """

    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]
    X = read_resize_flatten_image(folder, images_files[0], size)
    y = []
    y.append(get_file_number(images_files[0]))
    for i in range(1, len(images_files)):
        image_file = images_files[i]
        flatten = read_resize_flatten_image(folder, image_file, size)
        X = np.vstack((X, flatten))
        y.append(get_file_number(image_file))
    return X, np.asarray(y)


def read_resize_flatten_image(folder, image_file, size):
    file_path = os.path.join(folder, image_file)
    image = cv2.imread(file_path, 0)
    resized = cv2.resize(image, size)
    flatten = resized.flatten()
    return flatten


def get_file_number(image_file):
    name_divided = image_file.split('.')
    num = ''.join(filter(lambda i: i.isdigit(), name_divided[0]))
    return int(num)


def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    M = X.shape[0]
    idx_X = np.random.permutation(M)
    # y_rand = y[idx_X]
    N = int(M * p)
    training_idx, test_idx = idx_X[:N], idx_X[N:]
    Xtrain, Xtest = X[training_idx, :], X[test_idx, :]
    ytrain, ytest = y[training_idx], y[test_idx]
    return Xtrain, ytrain, Xtest, ytest


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """

    return np.mean(x, axis=0)


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """
    x_mean = get_mean_face(X)
    A = (X - x_mean)
    covariance = A.T @ A
    eigen_val,  eigen_vec = np.linalg.eigh(covariance)
    sorted_val = eigen_val[::-1][:k]
    sorted_vec = eigen_vec.T[::-1][:k]
    return sorted_vec.T, sorted_val


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        for i in range(self.num_iterations):
            self.weights = renormalize_weights(self.weights)
            wc = WeakClassifier(self.Xtrain, self.ytrain, self.weights)
            self.weakClassifiers.append(wc)
            wc.train()
            wk_results = [wc.predict(x) for x in self.Xtrain]
            correct = wk_results == self.ytrain
            normalized_weights = self.weights.copy()
            error_sum = normalized_weights[~correct].sum()
            alpha = np.log(np.sqrt((1-error_sum)/error_sum))
            self.alphas.append(alpha)
            if self.eps < error_sum:
                self.weights = update_weights(self.weights, self.ytrain, alpha, wk_results)
            else:
                break
        print("")

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        prediction = self.predict(self.Xtrain)
        matching = self.ytrain == prediction
        correct_count = np.count_nonzero(matching)
        total = len(matching)
        incorrect_count = total - correct_count
        return correct_count, incorrect_count

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        idx = 0
        wk_alpha_result = np.zeros(X.shape[0])
        for wk in self.weakClassifiers:
            wk_alpha_result += [self.alpha_prediction(wk, x, idx) for x in X]
            idx += 1
        prediction = np.sign(wk_alpha_result)
        return prediction

    def alpha_prediction(self, classifier, x, idx):
        return classifier.predict(x) * self.alphas[idx]


def renormalize_weights(weights):
    normalized = weights/weights.sum()
    return normalized


def update_weights(weights, label, alpha, predict):
    weights = weights * np.exp(-1*(label * alpha * predict))
    return weights


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        background = np.zeros(shape, np.uint8)
        org_row, org_column = self.position
        sq_height, sq_width = self.size
        center_row = org_row + (sq_height // 2)
        width = org_column + sq_width
        height = org_row + sq_height
        background[org_row:center_row, org_column:width] = 255
        background[center_row:height, org_column:width] = 126
        return background

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        background = np.zeros(shape, np.uint8)
        org_row, org_column = self.position
        sq_height, sq_width = self.size
        center_col = org_column + (sq_width // 2)
        width = org_column + sq_width
        height = org_row + sq_height
        background[org_row:height, org_column:center_col] = 255
        background[org_row:height, center_col:width] = 126
        return background

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        background = np.zeros(shape, np.uint8)
        org_row, org_column = self.position
        sq_height, sq_width = self.size
        center_upper_row = org_row + (sq_height // 3)
        center_lower_row = org_row + ((sq_height // 3) * 2)
        width = org_column + sq_width
        height = org_row + sq_height
        background[org_row:center_upper_row, org_column:width] = 255
        background[center_upper_row:center_lower_row, org_column:width] = 126
        background[center_lower_row:height, org_column:width] = 255
        return background

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        background = np.zeros(shape, np.uint8)
        org_row, org_column = self.position
        sq_height, sq_width = self.size
        center_left_col = org_column + (sq_width // 3)
        center_right_col = org_column + ((sq_width // 3)*2)
        width = org_column + sq_width
        height = org_row + sq_height
        background[org_row:height, org_column:center_left_col] = 255
        background[org_row:height, center_left_col:center_right_col] = 126
        background[org_row:height, center_right_col:width] = 255
        return background

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        background = np.zeros(shape, np.uint8)
        org_row, org_column = self.position
        sq_height, sq_width = self.size
        center_col = org_column + (sq_width // 2)
        center_row = org_row + (sq_height // 2)
        width = org_column + sq_width
        height = org_row + sq_height
        background[org_row:center_row, org_column:center_col] = 126
        background[center_row:height, org_column:center_col] = 255
        background[org_row:center_row, center_col:width] = 255
        background[center_row:height, center_col:width] = 126
        return background

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        org_row, org_column = self.position - np.array((1, 1))
        sq_height, sq_width = self.size
        width = org_column + sq_width
        height = org_row + sq_height

        if self.feat_type == (2, 1):  # two_horizontal
            center_row = org_row + (sq_height // 2)
            D = ii[height, width]
            B = ii[center_row, width]
            C = ii[height, org_column]
            A = ii[center_row, org_column]
            gray = D - B - C + A
            D2 = ii[center_row, width]
            B2 = ii[org_row, width]
            C2 = ii[center_row, org_column]
            A2 = ii[org_row, org_column]
            white = D2 - B2 - C2 + A2
            score = white - gray
        if self.feat_type == (1, 2):  # two_vertical
            center_col = org_column + (sq_width // 2)
            D = ii[height, width]
            B = ii[org_row, width]
            C = ii[height, center_col]
            A = ii[org_row, center_col]
            gray = D - B - C + A
            D = ii[height, center_col]
            B = ii[org_row, center_col]
            C = ii[height, org_column]
            A = ii[org_row, org_column]
            white = D - B - C + A
            score = white - gray
        if self.feat_type == (3, 1):  # three_horizontal
            center_upper_row = org_row + (sq_height // 3)
            center_lower_row = org_row + ((sq_height // 3) * 2)
            D = ii[height, width]
            B = ii[center_lower_row, width]
            C = ii[height, org_column]
            A = ii[center_lower_row, org_column]
            white1 = D - B - C + A
            D = ii[center_lower_row, width]
            B = ii[center_upper_row, width]
            C = ii[center_lower_row, org_column]
            A = ii[center_upper_row, org_column]
            gray = D - B - C + A
            D = ii[center_upper_row, width]
            B = ii[org_row, width]
            C = ii[center_upper_row, org_column]
            A = ii[org_row, org_column]
            white2 = D - B - C + A
            score = white1 - gray + white2

        if self.feat_type == (1, 3):  # three_vertical
            center_left_col = org_column + (sq_width // 3)
            center_right_col = org_column + ((sq_width // 3) * 2)
            D = ii[height, width]
            B = ii[org_row, width]
            C = ii[height, center_right_col]
            A = ii[org_row, center_right_col]
            white1 = D - B - C + A
            D = ii[height, center_right_col]
            B = ii[org_row, center_right_col]
            C = ii[height, center_left_col]
            A = ii[org_row, center_left_col]
            gray = D - B - C + A
            D = ii[height, center_left_col]
            B = ii[org_row, center_left_col]
            C = ii[height, org_column]
            A = ii[org_row, org_column]
            white2 = D - B - C + A
            score = white1 - gray + white2

        if self.feat_type == (2, 2):  # four_square
            center_col = org_column + (sq_width // 2)
            center_row = org_row + (sq_height // 2)
            D = ii[height, width]
            B = ii[center_row, width]
            C = ii[height, center_col]
            A = ii[center_row, center_col]
            gray_lower = D - B - C + A
            D = ii[center_row, width]
            B = ii[org_row, width]
            C = ii[center_row, center_col]
            A = ii[org_row, center_col]
            white_upper = D - B - C + A
            D = ii[height, center_col]
            B = ii[center_row, center_col]
            C = ii[height, org_column]
            A = ii[center_row, org_column]
            white_lower = D - B - C + A
            D = ii[center_row, center_col]
            B = ii[org_row, center_col]
            C = ii[center_row, org_column]
            A = ii[org_row, org_column]
            gray_upper = D - B - C + A
            score = white_upper - gray_lower + white_lower - gray_upper
        return score


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    integral_images = []
    for img in images:
        int_img = np.cumsum(np.cumsum(img, axis=0), axis=1)
        integral_images.append(int_img.astype(np.float))
    return integral_images


def update_weights_vj(weights, beta, correctness):
    for i, is_correct in enumerate(correctness):
        if is_correct:
            weights[i] = weights[i] * (beta**2)
    return weights


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei, sizej]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print(" -- select classifiers --")
        for i in range(num_classifiers):
            # TODO: Complete the Viola Jones algorithm
            weights = weights / weights.sum()
            vj_c = VJ_Classifier(scores, self.labels, weights)
            vj_c.train()
            vj_results = [vj_c.predict(x) for x in scores]
            self.classifiers.append(vj_c)
            matching = vj_results == self.labels
            beta = (vj_c.error/(1-vj_c.error))
            weights = update_weights_vj(weights, beta, matching)
            alpha = np.log10(1/beta)
            self.alphas.append(alpha)

    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.

        # Obtain the Haar feature id from clf.feature

        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'
        for idx, clf in enumerate(self.classifiers):
            feat_id = clf.feature
            feat_obj = self.haarFeatures[feat_id]
            for idx, x in enumerate(ii):
                score_val = feat_obj.evaluate(x)
                scores[idx, feat_id] = score_val

        result = []

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).
        alpha_sum = np.asarray(self.alphas).sum()
        alpha_threshold = 0.5 * alpha_sum
        for x in scores:
            # TODO
            weighted_pred = 0
            for idx, w_cl in enumerate(self.classifiers):
                prediction = w_cl.predict(x)
                weighted_pred += self.alphas[idx] * prediction
            if weighted_pred >= alpha_threshold:
                result.append(1)
            else:
                result.append(-1)
        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        # frame_out = image.copy()
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # height, width = 24, 24
        # highest_score = -1
        # # likely_window = gray_image
        # coord = [(0,0), (0,0)]
        # for h_idx in range(gray_image.shape[0]-height):
        #     for w_idx in range(gray_image.shape[1]-width):
        #         total_height = height+h_idx
        #         total_width = width+w_idx
        #         sub_window = gray_image[h_idx:total_height, w_idx:total_width]
        #         score = self.get_image_score(sub_window)
        #         if score >= highest_score:
        #             highest_score = score
        #             coord = [(w_idx, h_idx), (total_width, total_height)]
        # cv2.rectangle(frame_out, coord[0], coord[1], (0, 0, 255), 1)
        # cv2.imwrite("output/{}.png".format(filename), frame_out)

        frame_out = image.copy()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = 24, 24
        images = []
        images_coord = []
        for h_idx in range(gray_image.shape[0] - height):
            for w_idx in range(gray_image.shape[1] - width):
                total_height = height + h_idx
                total_width = width + w_idx
                sub_window = gray_image[h_idx:total_height, w_idx:total_width]
                images.append(sub_window)
                images_coord.append((h_idx, w_idx))
        prediction = self.predict(images)
        is_face_bool = np.array(prediction) != -1
        face_coord = np.asarray(images_coord)[is_face_bool,:]
        detected_ave = np.mean(face_coord, axis=0, dtype=int)
        detected_height = detected_ave[0] + height
        detected_width = detected_ave[1] + width
        cv2.rectangle(frame_out, (detected_ave[1], detected_ave[0]), (detected_width, detected_height) , (0, 0, 255), 1)
        cv2.imwrite("output/{}.png".format(filename), frame_out)




    def get_image_score(self, image):
        ii = convert_images_to_integral_images([image])
        scores = np.zeros((len(ii), len(self.haarFeatures)))
        for idx, clf in enumerate(self.classifiers):
            feat_id = clf.feature
            feat_obj = self.haarFeatures[feat_id]
            for idx, x in enumerate(ii):
                score_val = feat_obj.evaluate(x)
                scores[idx, feat_id] = score_val
        for x in scores:
            weighted_pred = 0
            for idx, w_cl in enumerate(self.classifiers):
                prediction = w_cl.predict(x)
                weighted_pred += self.alphas[idx] * prediction
        return weighted_pred


