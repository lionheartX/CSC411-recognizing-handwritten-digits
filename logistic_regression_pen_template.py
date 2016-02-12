import numpy as np
from check_grad import check_grad
from utils import *
from logistic import *
import matplotlib.pyplot as pyplot
from logistic_regression_template import run_check_grad

def run_logistic_regression_pen():
    #train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    #valid_inputs, valid_targets = load_test()

    N, M = train_inputs.shape
    
            
    hyperparameters = {
                    'learning_rate': 0.01,
                    'weight_regularization': 0.1,
                    'num_iterations': 1000
                 }

    regularizers = [0.001, 0.01, 0.1, 1.0]
    avg_ce_train = []
    avg_ce_valid = []
    avg_err_train = []
    avg_err_valid = []
    runs = 10
    
    for r in regularizers:
        hyperparameters['weight_regularization'] = r
        sum_ce_train = 0.0
        sum_ce_valid = 0.0
        sum_err_train = 0.0
        sum_err_valid = 0.0
        for i in xrange(runs):
            # Logistic regression weights
            weights = np.random.randn(M + 1, 1) /10

            # Verify that your logistic function produces the right gradient.
            # diff should be very close to 0.
            run_check_grad(hyperparameters)

            # Begin learning with gradient descent
            for t in xrange(hyperparameters['num_iterations']):

                # Find the negative log likelihood and its derivatives w.r.t. the weights.
                f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
        
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
            
            sum_ce_train += cross_entropy_train
            sum_ce_valid += cross_entropy_valid
            sum_err_train += (100.0 - frac_correct_train*100)
            sum_err_valid += (100.0 - frac_correct_valid*100)
            # print some stats
            #print "regularizer = %f" % r
            
            print ("regularizer = %s") % r
            print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
                   "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                        t+1, f / N, cross_entropy_train, frac_correct_train*100,
                        cross_entropy_valid, frac_correct_valid*100)
            print ""
        avg_ce_train.append(sum_ce_train/(runs*1.0))
        avg_ce_valid.append(sum_ce_valid/(runs*1.0))
        avg_err_train.append(sum_err_train/(runs*1.0))
        avg_err_valid.append(sum_err_valid/(runs*1.0))
    
    
    
    
    pyplot.title("Penalized Logistic Regression with mnist_train_small: Cross Entropy")
    pyplot.xlabel("Regularizer")
    pyplot.ylabel("Cross Entropy")
    pyplot.plot(regularizers, avg_ce_train, label = "training cross entropy")
    pyplot.plot(regularizers, avg_ce_valid, label = "validation cross entropy")
    pyplot.legend()
    pyplot.show()
    
    pyplot.title("Penalized Logistic Regression with mnist_train_small: Classification Error Rate")
    pyplot.xlabel("Regularizer")
    pyplot.ylabel("Classification Error Rate (%)")
    pyplot.plot(regularizers, avg_err_train, label = "Training Classification Error Rate")
    pyplot.plot(regularizers, avg_err_valid, label = "Validation Classification Error Rate")
    pyplot.legend()
    pyplot.show()
    
if __name__ == '__main__':
    run_logistic_regression_pen()
