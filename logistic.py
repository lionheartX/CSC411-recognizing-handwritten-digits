""" Methods for doing logistic regression."""

import numpy as np
import sys
from utils import sigmoid

def z_values(weights, data):
    """
    Compute the z values given weights and data.
    Z = W_transpose * X + b
    
    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        z:          :N x 1 vector of z values
    """
    N, M = data.shape
    modified_data = np.ones((N, M+1))
    modified_data[:,:-1] = data
    Z = np.dot(modified_data, weights)
    return Z
    
def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """
    y = sigmoid(z_values(weights, data))
    return y

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    N = targets.size
    # note this only works for binary labels
    y_labels = (y < 0.5).astype(np.int)
    
    if (N != y.size):
        print("evaluate error: mismatched targets and y")
        sys.exit(1)
        
    ce = -(np.sum(np.dot(targets.T,np.log(1.0 - y))+np.dot((1 - targets).T,np.log(y))))
    num_incorrect = np.sum(np.logical_xor(targets, y_labels))
    # since the output is either 1 or 0, we can simply xor
    frac_correct = 1.0 - num_incorrect / (N * 1.0)
    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """
    N, M = data.shape
    y = logistic_predict(weights, data)
    f = evaluate(targets, y)[0]
    modified_data = np.ones((N, M+1))
    modified_data[:,:-1] = data
    df = np.dot(modified_data.T, targets - (1.0 - y))
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:             N x 1 vector of probabilities.
    """
    
    #########################
    # variable declarations #
    #########################
    N, M = data.shape
    r = hyperparameters['weight_regularization']
    b = weights[-1]
    weights_no_bias = weights[:-1,:]
    
    #####################################
    # calculating cross entropy penalty #
    #         and gradient penalty      #
    #####################################
    pen_ce = sum((r/2.0*b*b) \
    + (r/2.0 * np.sum(weights_no_bias.T * weights_no_bias)) \
     - ((M + 1)/2.0*np.log(r/(2.0*np.pi))))
    pen_df = r*weights
    
    ###############################################
    # calculating penalized cross entropy penalty #
    #         and penalized entropy               #
    ###############################################
    y = logistic_predict(weights, data)
    f = evaluate(targets, y)[0] + pen_ce
    modified_data = np.ones((N, M+1))
    modified_data[:,:-1] = data
    df = np.dot(modified_data.T, targets - (1.0 - y)) + pen_df
    return f, df, y
