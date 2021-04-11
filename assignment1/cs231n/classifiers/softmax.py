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
    num_classes = W.shape[1]
    num_train = X.shape[0]

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


    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        ps = exp_scores / np.sum(exp_scores)
        loss += -1 * np.log(ps[y[i]])

        # gradient calcs
        for j in range(num_classes):
            dW[:,j] += ps[j]*X[i]
            if j == y[i]:
                dW[:,y[i]] += -1*X[i]

    # 1/N factor in front and regularization component
    loss /= num_train
    loss += reg * np.sum(W * W)

    # 1/N factor in front and regularization component
    dW /= num_train
    dW += 2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # build mask selecting only the correct classes (one-hot encodig of y_i)
    mask = np.eye(W.shape[1], dtype=bool)[y]

    S = X.dot(W)
    S -= np.max(S, axis=1)[:,np.newaxis]
    ES = np.exp(S)
    P = ES / np.sum(ES, axis=1)[:,np.newaxis]

    # compute loss
    loss = -1.0/num_train*np.sum(np.log(P[mask]))   +   reg * np.sum(W * W)
    #        -1/N            sum log(prob of y_i)   +   regularization part

    # gadiaent
    ones_yi = mask.astype(float)
    dW = 1.0/num_train * X.T.dot(P - ones_yi)    +   reg * 2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
