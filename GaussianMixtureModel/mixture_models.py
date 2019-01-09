from __future__ import division
import warnings
import numpy as np
import scipy as sp
from matplotlib import image
from random import randint
from scipy.misc import logsumexp
from helper_functions import image_to_matrix, matrix_to_image, \
                             flatten_image_matrix, unflatten_image_matrix, \
                             image_difference

warnings.simplefilter(action="ignore", category=FutureWarning)


def k_means_cluster(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """
    # TODO: finish this function
    # If initial_means is None, initial a random k means with RGB channels
    if initial_means is None:
        initial_means = np.random.random((k, 3))

    initial_means_copy = initial_means.copy()

    # Reshape to a two dimensional array with row as the pixels and columns as RGB channels
    img_reshaped = np.copy(image_values.reshape(-1, 3))

    #Loop over to find the converged centers
    while True:
        # Reshape the initial_means into 1 extra dimension for calculating the distance between each pixel and mean
        reshape_initial_mean = initial_means_copy.reshape(-1, 1, 3)

        # Calculate the different and then distance between image pixel and k means with broadcasting
        diff = reshape_initial_mean - img_reshaped

        # Euclidean disttance calculation along the RGB channels
        dist = np.linalg.norm(diff, axis = 2)

        # Label each pixel by the centers by finding the min distance between each pixel and k centers
        pixel_labels = np.argmin(dist, axis = 0)

        # Calculate new means based on the labeled pixels
        new_means = initial_means_copy.copy()

        # Loop over each centers
        for i in range(k):

            # Find pixels that belong the each cluster i
            curr_cluster = img_reshaped[np.where(pixel_labels == i)]

            # Calculate new center
            new_means[i] = np.sum(curr_cluster, axis = 0)/curr_cluster.shape[0]

        #print(initial_means.shape, new_means.shape)
        converged = True

        #check if convergence is reached
        for i in range(k):
            if not np.allclose(initial_means_copy[i], new_means[i], atol=1e-08):
                converged = False

        #update inital means with new means
        initial_means_copy = new_means

        if converged:
            break

    # Change the image to k means image
    for i in range(k):
        img_reshaped[np.where(pixel_labels == i)] = initial_means_copy[i]

    return img_reshaped.reshape(image_values.shape)


def default_convergence(prev_likelihood, new_likelihood, conv_ctr,
                        conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
                                abs(new_likelihood) <
                                abs(prev_likelihood) * 1.1)

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap


class GaussianMixtureModel:
    """
    A Gaussian mixture model
    to represent a provided
    grayscale image.
    """

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        self.image_matrix = image_matrix
        self.num_components = num_components
        if(means is None):
            self.means = np.zeros(num_components)
        else:
            self.means = means
        self.variances = np.zeros(num_components)
        self.mixing_coefficients = np.zeros(num_components)

    def joint_prob(self, val):
        """Calculate the joint
        log probability of a greyscale
        value within the image.

        params:
        val = float

        returns:
        joint_prob = float
        """
        # TODO: finish this
        #raise NotImplementedError()

        #Calculate log of gaussian density function
        log_prob = (-0.5) * np.log(2 * self.variances * np.pi) + (-1.0/(2 * self.variances)) * np.square(val - self.means)

        # Calculate the log prob with mixing coefficnets
        log_prob = np.log(np.sum(self.mixing_coefficients*np.exp(log_prob), axis = 0))

        return log_prob

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean to a random
        pixel's value (without replacement),
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

        NOTE: this should be called before
        train_model() in order for tests
        to execute correctly.
        """
        # TODO: finish this
        #raise NotImplementedError()
        self.variances = np.ones(self.num_components)
        self.mixing_coefficients = np.ones(self.num_components, dtype=np.float)/self.num_components
        self.means = np.random.choice(self.image_matrix.flatten(), self.num_components, replace = False)


    def train_model(self, convergence_function=default_convergence):
        """
        Train the mixture model
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and
        self.variances, plus
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.

        params:
        convergence_function = function, returns True if convergence is reached
        """
        # TODO: finish this
        #raise NotImplementedError()

        curr_likelihood = self.likelihood()

        conv_ctr = 0

        while True:

            # Expectation:
            # Gaussian distribution using current parameters
            Curr_Gaussian = 1.0/np.sqrt(2.0 * np.pi * self.variances) * np.exp((-0.5) * \
                            np.square(self.image_matrix.flatten().reshape(-1, 1) - \
                            self.means) / self.variances)

            # Calculate the mixed coeff times the gaussian distribution for each center
            Curr_Mix_Gaussian = self.mixing_coefficients * Curr_Gaussian


            # Sum up all the Gaussian center probabilities
            Sum_Mix_Gaussian = np.sum(Curr_Mix_Gaussian, axis = 1)
            
            # Calculate the responsibility
            curr_responsibility = np.divide(Curr_Mix_Gaussian, Sum_Mix_Gaussian.reshape(-1, 1))

            #print(self.image_matrix.shape, Curr_Gaussian.shape, Curr_Mix_Gaussian.shape, Sum_Mix_Gaussian.shape, curr_responsibility.shape)

            #Maximization:

            # Sum of log likelihood
            Sum_Responsibility = np.sum(curr_responsibility, axis = 0)

            #Calculate new mean values for the GMM
            self.means = np.sum(curr_responsibility * self.image_matrix.flatten().reshape(-1, 1), axis = 0)/Sum_Responsibility

            #Calculate new variances for the GMM
            self.variances = np.sum(curr_responsibility * \
                    np.square(self.image_matrix.flatten().reshape(-1, 1) - self.means), axis = 0)/Sum_Responsibility

            #Calculate new mixing_coeff for the GMM
            self.mixing_coefficients = Sum_Responsibility / self.image_matrix.flatten().shape[0]


            #print(self.image_matrix.flatten().shape[0], Sum_Responsibility)
            new_likelihood = self.likelihood()

            # Calculate convergence function
            conv_ctr, converged = convergence_function(curr_likelihood, new_likelihood, conv_ctr)

            # Refresh current likelihood
            curr_likelihood = new_likelihood

            #Check if converged
            if converged:
                break

    def segment(self):
        """
        Using the trained model,
        segment the image matrix into
        the pre-specified number of
        components. Returns the original
        image matrix with the each
        pixel's intensity replaced
        with its max-likelihood
        component mean.

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # TODO: finish this
        #raise NotImplementedError()
        img_shape = self.image_matrix.shape

        # Gaussian distribution using current parameters
        Curr_Gaussian = 1.0 / np.sqrt(2.0 * np.pi * self.variances) * np.exp((-0.5) * \
                        np.square(self.image_matrix.flatten().reshape(-1, 1) - \
                        self.means) / self.variances)

        # Calculate the mixed coeff times the gaussian distribution for each center
        Curr_Mix_Gaussian = self.mixing_coefficients * Curr_Gaussian

        # Sum up all the Gaussian center probabilities
        Sum_Mix_Gaussian = np.sum(Curr_Mix_Gaussian, axis=1)

        # Calculate the responsibility
        curr_responsibility = np.divide(Curr_Mix_Gaussian, Sum_Mix_Gaussian.reshape(-1, 1))

        # Calculate the labeling of each pixel by the most likely component
        labels = np.argmax(curr_responsibility, axis = 1)
        #print(curr_responsibility.shape, np.unique(labels), labels[:3],self.means.shape, self.num_components)
        new_img = self.image_matrix.copy()

        new_img = new_img.flatten()

        # Assign the pixel to its most likely center mean for segmentation
        for i in range(self.num_components):
            new_img[np.where(labels == i)] = self.means[i]

        self.image_matrix = new_img.reshape(img_shape)
        return self.image_matrix

    def likelihood(self):
        """Assign a log
        likelihood to the trained
        model based on the following
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), ln(sum((k=1 to K),
                                          mixing_k * N(x_n | mean_k,stdev_k))))

        returns:
        log_likelihood = float [0,1]
        """
        # TODO: finish this

        #Calculate log of gaussian density function
        #print(self.means.shape)
        #print(self.image_matrix.shape)
        log_likelihood = (-0.5) * np.log(2 * self.variances * np.pi) + (-1.0/(2 * self.variances)) * \
                   np.square(self.image_matrix.flatten().reshape(-1, 1) - self.means)
        #print((self.image_matrix.flatten().reshape(-1,1) - self.means).shape)
        # Calculate the log prob with mixing coefficnets
        log_likelihood = np.log(np.sum(self.mixing_coefficients*np.exp(log_likelihood), axis = 1))

        #print(np.sum(log_likelihood))
        return np.sum(log_likelihood)

    def best_segment(self, iters):
        """Determine the best segmentation
        of the image by repeatedly
        training the model and
        calculating its likelihood.
        Return the segment with the
        highest likelihood.

        params:
        iters = int

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # finish this

        #Initialize the best parameters, means, variances, and mix coeff
        final_means = None
        final_variances = None
        final_mix_coeff = None

        # Initialize best likelihood as -inf
        best_likelihood = float('-Inf')

        # Loop over all iterations required to find the best paramters
        for i in range(iters):
            # Initialize model
            self.initialize_training()
            self.train_model()
            curr_likelihood = self.likelihood()

            # Check if best likelihood is reached
            if best_likelihood < curr_likelihood:
                final_means = self.means
                final_variances = self.variances
                final_mix_coeff = self.mixing_coefficients
        # Update all parameters, means, variances, and mix coeff
        self.means = final_means
        self.variances = final_variances
        self.mixing_coefficients = final_mix_coeff

        return self.segment()


class GaussianMixtureModelImproved(GaussianMixtureModel):
    """A Gaussian mixture model
    for a provided grayscale image,
    with improved training
    performance."""

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean using some algorithm that
        you think might give better means to start with,
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
        [You can feel free to modify the variance and mixing coefficient
         initializations too if that works well.]
        """
        # TODO: finish this
        #raise NotImplementedError()


        #self.means = np.random.choice(self.image_matrix.flatten(), self.num_components, replace = False)

        max_pixel = np.max(self.image_matrix.flatten())
        min_pixel = np.min(self.image_matrix.flatten())
        #print(max_pixel, min_pixel)
        step = (max_pixel - min_pixel)/self.num_components

        for i in range(self.num_components):
            self.means[i] = (min_pixel + i * step + min_pixel + (i+1)*step)/2.0
        #print(self.means)

        # Using k means to find the best means

        initial_means_copy = self.means.copy()

        # Reshape to a two dimensional array with row as the pixels and columns as RGB channels
        img_reshaped = np.copy(self.image_matrix.flatten().reshape(-1, 1))

        while True:
            # Reshape the initial_means into 1 extra dimension for calculating the distance between each pixel and mean
            reshape_initial_mean = initial_means_copy.reshape(-1, 1, 1)

            # Calculate the different and then distance between image pixel and k means with broadcasting
            diff = reshape_initial_mean - img_reshaped
            dist = np.linalg.norm(diff, axis = 2)

            # Label each pixel by the centers by finding the min distance between each pixel and k means
            pixel_labels = np.argmin(dist, axis = 0)

            # Calculate new means based on the labeled pixels
            new_means = initial_means_copy.copy()
            for i in range(self.num_components):
                curr_cluster = img_reshaped[np.where(pixel_labels == i)]

                # Calculate new center
                new_means[i] = np.sum(curr_cluster)/curr_cluster.shape[0]
            #print(initial_means.shape, new_means.shape)
            converged = True
            for i in range(self.num_components):
                if not np.allclose(initial_means_copy[i], new_means[i], atol=1e-08):
                    converged = False

            initial_means_copy = new_means
            if converged:
                break

        self.means = initial_means_copy
        self.mixing_coefficients = np.ones(self.num_components)/self.num_components
        self.variances = np.ones(self.num_components)

def new_convergence_function(previous_variables, new_variables, conv_ctr,
                             conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:

    previous_variables = [numpy.ndarray[float]]
                         containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]]
                    containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    conv_ctr = int
    converged = boolean
    """
    # TODO: finish this function
    #raise NotImplementedError()

    # Check if all parameters, such as means, variances, and mix coeffs are within the +/-10%
    increase_convergence_ctr_means = (np.abs(previous_variables[0]) * 0.9 < np.abs(new_variables[0])).all() and \
                                      (np.abs(new_variables[0] < np.abs(previous_variables[0]) * 1.1)).all()
    increase_convergence_ctr_variances = (np.abs(previous_variables[1]) * 0.9 < np.abs(new_variables[1])).all() and  \
                                         (np.abs(new_variables[1]) < np.abs(previous_variables[1]) * 1.1).all()
    increase_convergence_ctr_mix_coeff = (np.abs(previous_variables[2]) * 0.9 < np.abs(new_variables[2])).all() and  \
                                         (np.abs(new_variables[2]) < np.abs(previous_variables[2]) * 1.1).all()

    if increase_convergence_ctr_means and increase_convergence_ctr_variances and \
            increase_convergence_ctr_mix_coeff:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap

class GaussianMixtureModelConvergence(GaussianMixtureModel):
    """
    Class to test the
    new convergence function
    in the same GMM model as
    before.
    """

    def train_model(self, convergence_function=new_convergence_function):
        # TODO: finish this function

        conv_ctr = 0

        while True:

            # Expectation:

            # Gaussian distribution using current parameters
            Curr_Gaussian = 1.0 / np.sqrt(2.0 * np.pi * self.variances) * np.exp((-0.5) * \
                           np.square(self.image_matrix.flatten().reshape(-1, 1) - \
                           self.means) / self.variances)

            # Calculate the mixed coeff times the gaussian distribution for each center
            Curr_Mix_Gaussian = self.mixing_coefficients * Curr_Gaussian

            # Sum up all the Gaussian center probabilities
            Sum_Mix_Gaussian = np.sum(Curr_Mix_Gaussian, axis=1)

            # Calculate the responsibility
            curr_responsibility = np.divide(Curr_Mix_Gaussian, Sum_Mix_Gaussian.reshape(-1, 1))

            # Current variables for improved convergence
            curr_variables = [self.means, self.variances, self.mixing_coefficients]

            # Maximization:

            # Sum of log likelihood
            Sum_Responsibility = np.sum(curr_responsibility, axis=0)

            # Calculate new mean values for the GMM
            self.means = np.sum(curr_responsibility * self.image_matrix.flatten().reshape(-1, 1),
                                axis=0) / Sum_Responsibility

            # Calculate new variances for the GMM
            self.variances = np.sum(curr_responsibility * \
                                    np.square(self.image_matrix.flatten().reshape(-1, 1) - self.means),
                                    axis=0) / Sum_Responsibility

            # Calculate new mixing_coeff for the GMM
            self.mixing_coefficients = Sum_Responsibility / self.image_matrix.flatten().shape[0]

            # New variables for improved convergence
            new_variables = [self.means, self.variances, self.mixing_coefficients]

            # Calculate convergence function
            conv_ctr, converged = convergence_function(curr_variables, new_variables, conv_ctr)

            if converged:
                break


def bayes_info_criterion(gmm):
    # TODO: finish this function
    #raise NotImplementedError()
    BIC = 3.0 * gmm.num_components * np.log(gmm.image_matrix.flatten().shape[0]) - 2.0 * gmm.likelihood()
    return BIC

def BIC_likelihood_model_test():
    """Test to compare the
    models with the lowest BIC
    and the highest likelihood.

    returns:
    min_BIC_model = GaussianMixtureModel
    max_likelihood_model = GaussianMixtureModel

    for testing purposes:
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563, 0.964706]
    ]
    """
    # TODO: finish this method
    #raise NotImplementedError()

    # Use the given means
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563, 0.964706]
    ]

    # Image to matrix
    img = image_to_matrix('images/self_driving.png', grays= True)

    print(img.shape)

    # Setting up the initial values for the best k, best BIC and likelihood
    best_BIC_k = -1
    best_likelihood_k = -1

    min_BIC = float('Inf')
    max_likelihood = float('-Inf')

    min_BIC_model = None
    max_likelihood_model = None


    # Initialize k
    k = 2
    # Loop through k from 2 to 7 and loop through all means
    for mean in comp_means:
        gmm = GaussianMixtureModel(img.copy(), k)
        gmm.initialize_training()
        gmm.means = np.copy(mean)
        gmm.num_components = k
        gmm.train_model()
        BIC = bayes_info_criterion(gmm)
        likelihood = gmm.likelihood()

        # Check if current BIC is smaller than min_BIC
        if min_BIC > BIC:
            min_BIC = BIC
            best_BIC_k = k
            min_BIC_model = gmm


        # Check if current likelihood is larger than max_likelihood
        if max_likelihood < likelihood:
            max_likelihood = likelihood
            best_likelihood_k = k
            max_likelihood_model = gmm

        k += 1

    #print out the result of the best k
    print(best_BIC_k, best_likelihood_k)

    return min_BIC_model, max_likelihood_model


def BIC_likelihood_question():
    """
    Choose the best number of
    components for each metric
    (min BIC and maximum likelihood).

    returns:
    pairs = dict
    """
    # TODO: fill in bic and likelihood
    #raise NotImplementedError()
    # Updat the bic and likelihood with the results from BIC_likelihood_model_test()
    bic = 2
    likelihood = 6
    pairs = {
        'BIC': bic,
        'likelihood': likelihood
    }
    return pairs

def return_your_name():
    # return your name
    # TODO: finish this
    #raise NotImplemented()
    return "Xiangnan He"

def bonus(points_array, means_array):
    """
    Return the distance from every point in points_array
    to every point in means_array.

    returns:
    dists = numpy array of float
    """
    # TODO: fill in the bonus function
    # REMOVE THE LINE BELOW IF ATTEMPTING BONUS
    #raise NotImplementedError()

    #X, n = points_array.shape
    #Y, n = means_array.shape

    #print(X, Y, n)
    #Calculate difference table
    diff = points_array.reshape(-1, 1, n) - means_array
    dists = np.linalg.norm(diff, axis = 2)
    #print(dists)
    return dists


