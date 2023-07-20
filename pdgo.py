from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from sklearn.linear_model import LogisticRegression
import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import check_array, check_random_state
from collections import Counter



__author__ = 'taoll'



class PDGO(object):
    """
    Oversampling parent class with the main methods required by scikit-learn:
    fit, transform and fit_transform
    """

    def __init__(self,
                 alpha=0.5,
                 beta=0.2,
                 k=5,
                 random_state=None,
                 verbose=True):


        '''
        
        :param alpha: The adjustment factor between prior probability and purity.
        :param beta: Gaussian sampling variance adjustment factor.
        :param k: The setting of the k value in the k-NN algorithm.
        :param random_state: The setting of the random seed.
        :param verbose: Whether to print exceptions.
        '''
        self.alpha=alpha
        self.beta = beta
        self.k = k
        self.random_state = random_state
        self.verbose = verbose
        self.clstats = {}
        self.num_new = 0
        self.index_new = []


    def fit(self, X, y):
        """
        Class method to define class populations and store them as instance
        variables. Also stores majority class label
        """

        self.X = check_array(X)
        self.y = np.array(y)
        self.random_state_ = check_random_state(self.random_state)
        self.unique_classes_ = np.unique(self.y)
        # Counting class labels.
        classes = np.unique(self.y)
        # Calculate the data volume for each class label.
        sizes = np.array([sum(y == c) for c in classes])
        indices = np.argsort(sizes)[::-1]
        # Sort the class labels in descending order of their data volume.
        self.unique_classes_ = classes[indices]


        # Initialize the total count of all classes, which is set to 0 by default.
        for element in self.unique_classes_:
            self.clstats[element] = 0

        # Calculate the count of each class.
        for element in self.y:
            self.clstats[element] += 1

        # Find majority class
        v = list(self.clstats.values())
        k = list(self.clstats.keys())
        self.maj_class_ = k[v.index(max(v))]

        if self.verbose:
            print(
                'Majority class is %s and total number of classes is %s'
                % (self.maj_class_, len(self.unique_classes_)))


    def fit_sample(self, X, y):
        '''
        todo：Entry call function.
        :param X: Original sample features.
        :param y: Sample labels.
        :return: The generated samples after merging with the original samples.
        '''

        self.fit(X, y)
        self.new_X, self.new_y = self.oversample()
        # The combined generated samples and original samples after sampling.
        self.new_X = np.concatenate((self.new_X, self.X), axis=0)
        self.new_y = np.concatenate((self.new_y, self.y), axis=0)
        return self.new_X, self.new_y

    def generate_samples(self, x, knns, knnLabels, cl):
        '''
        
        :param x:  Indices of the minority class in the original matrix.
        :param knns: Indices of the K nearest samples for each minority class.
        :param knnLabels: Labels of the K nearest samples for each minority class.
        :param cl: The labels of that class.
        :return: 
        '''
        # List to store synthetically generated samples and their labels
        new_data = []
        new_labels = []
        for ind, elem in enumerate(x):
            # knns[ind][1:] excluding the first one (which is the minority class sample itself)，knnLabelsp[ind][+1]
            #The indices of the minority class samples.
            min_knns = [ele for index,ele in enumerate(knns[ind][1:])
                         if knnLabels[ind][index+1] == cl]

            if not min_knns:
                continue

            # generate gi synthetic examples for every minority example
            for i in range(0, int(self.gi[ind])):
                # randi holds an integer to choose a random minority kNNs
                randi = self.random_state_.random_integers(
                    0, len(min_knns) - 1)
                #The variance of the Gaussian distribution is determined by multiplying the reciprocal of the prior probability by an adjustment factor. If the probability is larger, the variance of the Gaussian distribution will be smaller, resulting in a larger sampling range.
                std = self.beta * (1 / self.priori[ind])
                r = np.random.normal(loc=0, scale= std)
                # Generate samples.
                si = self.X[elem] +  (self.X[min_knns[randi]] - self.X[elem]) * r
                # Add to the newly generated samples.
                new_data.append(si)
                new_labels.append(self.y[elem])
                self.num_new += 1

        return(np.asarray(new_data), np.asarray(new_labels))

    def loistic_prob(self,X,y):
        '''
        Compute the probability of each sample belonging to the corresponding class using logistic regression.
        :param X: Original sample features.
        :param y: Sample labels.
        :return: 

        lr = LogisticRegression(max_iter=2000)
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)

        y = LabelEncoder().fit_transform(y)
        lr_model = lr.fit(X, y)
        proba = lr_model.predict_proba(X)

        priori = [pi[label] for label, pi in zip(y, proba)]
        return np.array(priori)



    def oversample(self):
        """
        Preliminary calculations before generation of
        synthetic samples. Calculates and stores as instance
        variables: img_degree(d),G,ri,gi as defined by equations
        [1],[2],[3],[4] in the original paper
        """
        try:
            # Checking if variable exists, i.e. if fit() was called
            self.unique_classes_ = self.unique_classes_
        except:
            raise RuntimeError("You need to fit() before applying tranform(),"
                               "or simply fit_transform()")
        #Store the newly generated samples.
        new_X = np.zeros([1, self.X.shape[1]])
        new_y = np.zeros([1])
        # Calculate the prior probabilities based on logistic regression.
        self.priori = self.loistic_prob(self.X,self.y)
        # During the iteration loop, we need the minority class samples. However, 
        #since unique_classes_ is sorted in descending order by quantity, 
        #we do not need to sample from the first class.
        for i in range(1, len(self.unique_classes_)):
            minority_label = self.unique_classes_[i]

            # G represents the number of samples to be generated for the minority class, 
            self.G = self.clstats[self.maj_class_] - self.clstats[minority_label]

            # PPGO is built upon eucliden distance so p=2 default
            self.nearest_neighbors_ = NearestNeighbors(n_neighbors=self.k + 1)
            self.nearest_neighbors_.fit(self.X)

            #Obtain the indices of the minority class samples.
            minx = [ind for ind, exam in enumerate(self.X) if self.y[ind] == minority_label]

            # Calculate the k-nearest neighbors for each minority class sample.
            knn = self.nearest_neighbors_.kneighbors(self.X[minx], return_distance=False)

            #Obtain the labels of the k-nearest neighbors for each minority class sample.
            knnLabels = self.y[knn.ravel()].reshape(knn.shape)
            #Count the number of samples for each label.
            tempdi = [Counter(i) for i in knnLabels]

            #Calculate the proportion of the minority class in the K nearest neighbors.
            ratio =np.array([(i[minority_label]-1) / float(self.k) for i in tempdi])
            #Combine the prior probability and purity.
            self.ri = np.array(self.alpha*self.priori[minx]+(1-self.alpha)* ratio)
            # Normalize the weights.
            if np.sum(self.ri):
                self.ri = self.ri / np.sum(self.ri)

            # Round the weights to the nearest integer.
            self.gi = np.rint(self.ri * self.G)

            # Generate new synthetic samples for the current minority class.
            temp_new_X, curr_new_y = self.generate_samples(minx, knn, knnLabels, minority_label)

            # Concatenate the newly generated synthetic samples for the minority class.
            if len(temp_new_X):
                new_X = np.concatenate((new_X, temp_new_X), axis=0)
            if len(curr_new_y):
                new_y = np.concatenate((new_y, curr_new_y), axis=0)
        # New samples are concatenated in the beggining of the X,y arrays
        # index_new contains the indiced of artificial examples
        self.index_new = [i for i in range(0,self.num_new)]
        return(new_X[1:], new_y[1:])



if __name__ == '__main__':
    X = np.array(
        [
            [0.11622591, -0.0317206],
            [0.77481731, 0.60935141],
            [1.25192108, -0.22367336],
            [0.53366841, -0.30312976],
            [1.52091956, -0.49283504],
            [-0.28162401, -2.10400981],
            [0.83680821, 1.72827342],
            [0.3084254, 0.33299982],
            [0.70472253, -0.73309052],
            [0.28893132, -0.38761769],
            [1.15514042, 0.0129463],
            [0.88407872, 0.35454207],
            [1.31301027, -0.92648734],
            [-1.11515198, -0.93689695],
            [-0.18410027, -0.45194484],
            [0.9281014, 0.53085498],
            [-0.14374509, 0.27370049],
            [-0.41635887, -0.38299653],
            [0.08711622, 0.93259929],
            [1.70580611, -0.11219234],
        ]
    )
    y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    resample_X,resample_y = PDGO().fit_sample(X,y)
    print(resample_X,len(resample_y[resample_y==0]))
    print(len(resample_y))


    pass
