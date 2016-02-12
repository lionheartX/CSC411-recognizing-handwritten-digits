import numpy as np
from l2_distance import l2_distance
import matplotlib.pyplot as pyplot
from utils import *

def run_knn(k, train_data, train_labels, valid_data):
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples, 
          and M is the number of features per example.

    Inputs:
        k:            The number of neighbours to use for classification 
                      of a validation example.
        train_data:   The N_TRAIN x M array of training
                      data.
        train_labels: The N_TRAIN x 1 vector of training labels
                      corresponding to the examples in train_data 
                      (must be binary).
        valid_data:   The N_VALID x M array of data to
                      predict classes for.

    Outputs:
        valid_labels: The N_VALID x 1 vector of predicted labels 
                      for the validation data.
    """

    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:,:k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # note this only works for binary labels
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1,1)

    return valid_labels

if __name__ == '__main__':
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()

    num_valid = valid_targets.size
    classification_rates_list = []
    
    ks = [1, 3, 5, 7, 9]
    for k in ks:
        valid_labels = run_knn(k, train_inputs, train_targets, valid_inputs)
        num_incorrect = np.sum(np.logical_xor(valid_labels, valid_targets))
        # since the output is either 1 or 0, we can simply xor
        classification_rate = 1.0 - num_incorrect / (num_valid * 1.0)
        classification_rates_list.append(classification_rate)
        
    pyplot.title("2.1 - kNN classification rate") 
    pyplot.xlabel("k")
    pyplot.ylabel("classification rate")
    pyplot.plot(ks, classification_rates_list)
    pyplot.show()
    