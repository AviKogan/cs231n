from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        f = X[i].dot(W)
        f -= f.max() # normalization trick
        exp_f = np.exp(f)
        f_sum = np.sum(exp_f)

        f_yi = exp_f[y[i]]

        loss += (-1) * np.log(f_yi / f_sum)

        dW[:, y[i]] += (-1) * ((f_sum - f_yi) / f_sum) * X[i]

        for j in range(num_classes):
            if j == y[i]:
                continue
            
            dW[:, j] += (exp_f[j] / f_sum) * X[i]

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    f = X.dot(W)
    f -= f.max() # normalization trick
    exp_f = np.exp(f)
    exp_f_sums = exp_f.sum(axis = 1)

    exp_f_yi = exp_f[np.arange(num_train), y]
    
    #loss
    loss = exp_f_yi / exp_f_sums
    loss = (-1) * np.log(loss).sum() / num_train + reg * np.sum(W * W)

    #gradient
    #first the derivative of CE according to exp^W_j
    dW_no_inner = np.divide(exp_f, exp_f_sums.reshape(num_train, 1))
    dW_no_inner[range(num_train), y] = - (exp_f_sums - exp_f_yi) / exp_f_sums

    #now multipling by the innned derivative
    dW = X.T.dot(dW_no_inner)
    dW /= num_train
    dW += 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
