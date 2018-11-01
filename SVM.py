# coding: utf-8
import numpy as np

np.random.seed(420)


class SVMHelper:
    """ SVM model """

    @staticmethod
    def init_weights(X, a, b):
        """ Initialization of weights"""
        _, num_features = X.shape
        return a + (b - a) * np.random.random(num_features)

    @staticmethod
    def dot_product(x, weights):
        """ Dot calculation"""
        return sum(x * weights)

    @staticmethod
    def sign(x):
        """ Sign class calculation """
        return 1 if x >= 0.0 else -1

    @classmethod
    def predict(cls, x, weights):
        """ Prediction """
        score = cls.dot_product(x, weights)
        label = cls.sign(score)

        return label, score

    @staticmethod
    def hinge_loss(predicted_value, true_value):
        """ Hinge loss function """
        return max(0.0, 1.0 - predicted_value * true_value)

    @classmethod
    def hinge_loss_dataset(cls, predicted_values, true_values):
        """ Hinge loss function for dataset """
        result = 1 - predicted_values * true_values
        result[result < 0] = 0
        return sum(result) / len(predicted_values)

    @classmethod
    def hinge_loss_in_point(cls, X, y, weights):
        """ Hinge loss function for dataset for weights"""
        predicted_values = np.sum(X * weights, axis=1)
        return cls.hinge_loss_dataset(predicted_values, y)

    @classmethod
    def regul_loss_in_point(cls, C):
        """ Regularization for coefficient """
        def loss_fun(X, y, weights):
            """ Loss function for coefficient """
            result = cls.hinge_loss_in_point(X, y, weights) + 0.5 / C * np.linalg.norm(weights)
            return result

        return loss_fun

    @classmethod
    def gradient(cls, loss_fun, X, y, model_weights, w_delta=0.01, empties=None):
        """ Gradient descent """
        current_loss = loss_fun(X, y, model_weights)
        weights_delta = model_weights[:]
        gradient = []

        if empties is None:
            empties = []

        for coord in range(len(model_weights)):
            if coord in empties:
                gradient.append(0)
                continue
            weights_delta[coord] += w_delta
            delta_loss = loss_fun(X, y, weights_delta)
            deriv = (delta_loss - current_loss) / w_delta
            gradient.append(deriv)
            weights_delta[coord] -= w_delta

        return np.array(gradient)

    @classmethod
    def gradient_descent(cls, loss_fun, X, y, init_weights, learning_rate, empties=[], verbose=True):
        """ Gradient Descent """
        cur_weights = init_weights[:]
        counter = 1

        loss_before = loss_fun(X, y, cur_weights)

        while True:
            happy_indexes = np.random.randint(0, X.shape[0], 1000)

            X_mod = X[happy_indexes]
            y_mod = y[happy_indexes]

            grad = cls.gradient(loss_fun, X_mod, y_mod, cur_weights, empties=empties)
            cur_weights -= learning_rate * grad
            loss_after = loss_fun(X, y, cur_weights)

            if loss_after < loss_before:
                learning_rate *= 1.05
            elif loss_after >= loss_before:
                learning_rate /= 2

            if verbose:
                print("Iter: %i, loss_value: %f, rate: %f" % (counter, loss_after, learning_rate))
            counter += 1

            if abs(loss_after - loss_before) < 0.0001 or learning_rate < 0.00001 or counter > 500:
                break
            else:
                loss_before = loss_after

        return cur_weights
