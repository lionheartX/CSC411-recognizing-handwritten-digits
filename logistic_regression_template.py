import numpy as np
from check_grad import check_grad
from utils import *
from logistic import *
import matplotlib.pyplot as pyplot

def run_logistic_regression():
    train_inputs, train_targets = load_train()
    #train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    #valid_inputs, valid_targets = load_test()

    N, M = train_inputs.shape

    hyperparameters = {
                    'learning_rate': 0.01,
                    'weight_regularization': 0.1,
                    'num_iterations': 1000
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.randn(M + 1, 1) /10

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)
    ce_train = []
    ce_valid = []

    # Begin learning with gradient descent
    for t in xrange(hyperparameters['num_iterations']):

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        
        # print some stats
        print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
               "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                   t+1, f / N, cross_entropy_train, frac_correct_train*100,
                   cross_entropy_valid, frac_correct_valid*100)
        ce_train.append(cross_entropy_train)
        ce_valid.append(cross_entropy_valid)   
          
    pyplot.title("Logistic Regression with mnist_train")
    pyplot.xlabel("Training Iteration")
    pyplot.ylabel("Cross Entropy")
    pyplot.plot(xrange(1, hyperparameters['num_iterations'] + 1), ce_train, label = "training cross entropy")
    pyplot.plot(xrange(1, hyperparameters['num_iterations'] + 1), ce_valid, label = "validation cross entropy")
    pyplot.legend()
    pyplot.show()

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    #print "diff =", diff

if __name__ == '__main__':
    run_logistic_regression()
