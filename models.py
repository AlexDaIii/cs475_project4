import numpy as np


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y, **kwargs):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class Useless(Model):

    def __init__(self):
        super().__init__()
        self.reference_example = None
        self.reference_label = None

    def fit(self, X, y, **kwargs):
        self.num_input_features = X.shape[1]
        # Designate the first training example as the 'reference' example
        # It's shape is [1, num_features]
        self.reference_example = X[0, :]
        # Designate the first training label as the 'reference' label
        self.reference_label = y[0]
        self.opposite_label = 1 - self.reference_label

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        # Perhaps fewer features are seen at test time than train time, in
        # which case X.shape[1] < self.num_input_features. If this is the case,
        # we can simply 'grow' the rows of X with zeros. (The copy isn't
        # necessary here; it's just a simple way to avoid modifying the
        # argument X.)
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        # Compute the dot products between the reference example and X examples
        # The element-wise multiply relies on broadcasting; here, it's as if we first
        # replicate the reference example over rows to form a [num_examples, num_input_features]
        # array, but it's done more efficiently. This forms a [num_examples, num_input_features]
        # sparse matrix, which we then sum over axis 1.
        dot_products = X.multiply(self.reference_example).sum(axis=1)
        # dot_products is now a [num_examples, 1] dense matrix. We'll turn it into a
        # 1-D array with shape [num_examples], to be consistent with our desired predictions.
        dot_products = np.asarray(dot_products).flatten()
        # If positive, return the same label; otherwise return the opposite label.
        same_label_mask = dot_products >= 0
        opposite_label_mask = ~same_label_mask
        y_hat = np.empty([num_examples], dtype=np.int)
        y_hat[same_label_mask] = self.reference_label
        y_hat[opposite_label_mask] = self.opposite_label
        return y_hat


class LambdaMeans(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        self.centroids = None  # a k*num_features matrix - these are the means in k means
        self.num_clusters = 1  # this is the number of clusters k
        self.num_features = None  # number of features
        self.num_examples = None  # these are the number of examples
        self.lmda = None  # this is the lambda value
        self.num_iter = None  # this is the number of iterations
        self.clusters = None  # these are the assignments X into the clusters
        pass

    def fit(self, X, _, **kwargs):
        """  Fit the lambda means model  """
        assert 'lambda0' in kwargs, 'Need a value for lambda'
        assert 'iterations' in kwargs, 'Need the number of EM iterations'
        lambda0 = kwargs['lambda0']
        iterations = kwargs['iterations']

        # TODO: Write code to fit the model.  NOTE: labels should not be used here.
        self.centroids = np.mean(X, 0)  # first centroid is the mean of the data

        self.num_features = np.size(X, 1)
        self.num_examples = np.size(X, 0)
        self.clusters = [[-1]]  # list works the closest to vector in cpp

        self.num_iter = iterations
        if lambda0 <= 0:  # checks if lambda is valid - if not then initialize lambda for them
            self.lmda = self.initilize_lambda(X)
        else:
            self.lmda = lambda0

        self.train(X)

    def predict(self, X):
        # TODO: Write code to make predictions.
        X = X.todense()

        num_examples, num_input_features = X.shape
        # if num input features less, then append zeros
        if num_input_features < self.num_features:
            temp = np.zeros((num_examples, self.num_features - num_input_features))
            X = np.append(X, temp, 1)
        # if the num input features greater, get rid of some features
        if num_input_features > self.num_features:
            X = X[:, :self.num_features]

        y = np.zeros((num_examples, 1))

        for i in range(num_examples):
            y[i] = self.find_closest_for_predict(X[i])
        return y

    def find_closest_for_predict(self, point):
        x = np.linalg.norm(self.centroids - point, axis=1)
        closest = np.argmin(x, axis=0)
        return closest

    def calculate_distance(self, x1, x2):
        # TODO: VECTORIZE THIS
        """
        Takes in two vectors and calculates the distance
        :param x1: First vector
        :param x2: Second vector
        :return: the distance to vector
        """
        return np.linalg.norm(x1 - x2, axis=1)

    def update_centroids(self, X):
        """
        For all centroids update it to the mean of the cluster
        :param X: the data
        """
        for i in range(self.num_clusters):
            self.centroids[i, :] = np.mean(X[self.clusters[i][1:]], axis=0)
        pass

    def find_closest_cluster(self, point):
        """
        Gives the idx - row - of the closest centroid
        :param ck: If we are checking lambda
        :param point: the point to find the closest centroid to
        :return: the idx of the closest centroid
        """
        x = np.linalg.norm(self.centroids - point, axis=1)
        min_distance = np.min(x, axis=0)
        closest = np.argmin(x, axis=0)
        return self.check_lambda(min_distance, point, closest)

    def check_lambda(self, min_distance, point, closest):
        """
        Checks if the min distance is less than lambda
        :param min_distance: the minimum distance
        :param point: point to return
        :param closest: closest idx
        :return: the idx fo closest cluster
        """
        if min_distance < self.lmda:
            return closest
        else:
            self.create_new_centroid(point)
            return self.num_clusters - 1

    def create_new_centroid(self, point):
        """
        If the point is too far from a centroid, then create a new centroid at that point
        :param point: the point that was too far - create a new centroid at that point
        """
        self.num_clusters += 1
        self.centroids = np.concatenate((self.centroids, point), axis=0)
        self.clusters.append([-1])
        pass

    def initilize_lambda(self, X):
        """
        Initializes the lambda to be the average distance to the initial centroid
        :param X: the data
        :return: the lambda
        """
        lmd_wo = np.sum(np.linalg.norm(np.subtract(X, self.centroids), axis=1))  # default lambda
        return np.divide(lmd_wo, self.num_examples)

    def clear_assignments(self):
        """
        Clears assignments to the clusters
        """
        for i in range(self.num_clusters):
            self.clusters[i] = [-1]

    def train(self, X):
        for iteration in range(self.num_iter):
            for i in range(self.num_examples):
                cluster_assigment = self.find_closest_cluster(X[i])
                self.clusters[cluster_assigment].append(i)
            self.update_centroids(X)
            self.clear_assignments()
        pass


class StochasticKMeans(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        self.centroids = None  # a k*num_features matrix - these are the means in k means
        self.num_clusters = None  # this is the number of clusters k
        self.num_features = None  # number of features
        self.num_examples = None  # these are the number of examples
        self.beta = None  # this is the beta value
        self.num_iter = None  # this is the number of iterations
        self.clusters = None  # these are the assignments X into the clusters
        pass

    def fit(self, X, _, **kwargs):
        assert 'num_clusters' in kwargs, 'Need the number of clusters (K)'
        assert 'iterations' in kwargs, 'Need the number of EM iterations'
        num_clusters = kwargs['num_clusters']
        iterations = kwargs['iterations']
        # TODO: Write code to fit the model.  NOTE: labels should not be used here.

        self.num_features = np.size(X, 1)
        self.num_examples = np.size(X, 0)
        self.num_clusters = num_clusters
        self.num_iter = iterations
        self.beta = 2

        self.init_centroids(X)
        self.train(X)

    def init_centroids(self, X):
        if self.num_clusters == 1:
            self.centroids = np.mean(X, 0)  # first centroid is the mean of the data
            self.clusters = [[-1]]
        elif self.num_clusters >= 2:
            minimia = np.min(X, axis=0)
            maximia = np.max(X, axis=0)
            if self.num_clusters == 2:
                self.centroids = np.zeros((2, self.num_features))
                self.centroids[0,:] = minimia
                self.centroids[1,:] = maximia
                self.clusters = [[-1],[-1]]
            elif self.num_clusters == 3:
                self.centroids = np.zeros((3, self.num_features))
                self.centroids[0, :] = minimia
                self.centroids[1, :] = maximia
                self.centroids[2, :] = np.divide((maximia + minimia), 2)
                self.clusters = [[-1],[-1],[-1]]
            elif self.num_clusters == 4:
                self.centroids = np.zeros((4, self.num_features))
                self.centroids[0, :] = minimia
                self.centroids[1, :] = maximia
                self.centroids[2, :] = np.divide(maximia, 3) + np.multiply(minimia, 2/3)
                self.centroids[2, :] = np.divide(minimia, 3) + np.multiply(maximia, 2/3)
                self.clusters = [[-1], [-1], [-1], [-1]]
        pass

    def predict(self, X):
        # TODO: Write code to make predictions.
        X = X.todense()

        num_examples, num_input_features = X.shape
        # if num input features less, then append zeros
        if num_input_features < self.num_features:
            temp = np.zeros((num_examples, self.num_features - num_input_features))
            X = np.append(X, temp, 1)
        # if the num input features greater, get rid of some features
        if num_input_features > self.num_features:
            X = X[:, :self.num_features]

        y = np.zeros((num_examples, 1))

        for i in range(num_examples):
            y[i] = self.find_closest_cluster(X[i])
        return y

    def train(self, X):
        for iteration in range(self.num_iter):
            for i in range(self.num_examples):
                cluster_assigment = self.find_closest_cluster(X[i])
                self.clusters[cluster_assigment].append(i)
            self.update_centroids(X)
            self.clear_assignments()
        pass

    def find_closest_cluster(self, point):
        """
        Gives the idx - row - of the closest centroid
        :param point: the point to find the closest centroid to
        :return: the idx of the closest centroid
        """
        x = np.linalg.norm(self.centroids - point, axis=1)
        closest = np.argmin(x, axis=0)
        return closest

    def update_centroids(self, X):
        """
        For all centroids update it to the mean of the cluster
        :param X: the data
        """
        for i in range(self.num_clusters):
            self.centroids[i, :] = self.calculate_prob(X)
        pass

    def clear_assignments(self):
        """
        Clears assignments to the clusters
        """
        for i in range(self.num_clusters):
            self.clusters[i] = [-1]

    def calculate_prob(self, X):
        dist = np.linalg.norm(X - self.centroids[0, :], axis=1)
        inner = np.multiply(-self.beta, dist)
        inner = np.divide(inner, np.mean(dist))
        numerator = np.exp(inner)
        denom = np.sum(inner)
        return np.divide(numerator, denom)
