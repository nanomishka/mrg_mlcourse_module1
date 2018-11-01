# coding: utf-8
import time
import numpy as np


class TimeHelper:
    """ Time execution helper """
    def __init__(self, title):
        self.title = title
        self.start_time = time.time()

    def finish(self):
        """ Result output """
        elapsed_time = time.time() - self.start_time
        print('{} for {} sec'.format(self.title, round(elapsed_time, 3)))


class DataHelper:
    """ Data format helper """

    @staticmethod
    def normalize(X):
        """ Normalization """
        norms = np.amax(X, 0)
        norms[norms == 0] = 1
        return X / norms

    @staticmethod
    def bias(X):
        """ Bias adding """
        num_examples, _ = X.shape
        return np.hstack((np.ones((num_examples, 1)), X))
