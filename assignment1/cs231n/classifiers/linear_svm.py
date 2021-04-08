from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # 3073 x 10 zeros

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += 1*X[i]
                dW[:,y[i]] += -1*X[i]

    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # 1/N factor in front
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW += 2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    num_train = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    S = X.dot(W)  # scores (N,C)
    # build mask selecting only the correct classes (one-hot encodig of y_i)
    mask = np.eye(W.shape[1], dtype=bool)[y] 

    # correct scores which we'll be subtracting from all other
    correct_scores_vec = np.sum(np.where(mask, S, 0), axis=1)
    correct_scores = correct_scores_vec[:,np.newaxis]   # broadcasting-ready vec

    # compute margins
    M = S - correct_scores + 1    # margins   (N,C)
    M[mask] = 0
    pM = np.where(M>0, M, 0)      # positive marings

    # compute loss
    loss = 1.0/num_train * np.sum(pM)   +   reg * np.sum(W * W)
    #             maring conributions   +   regularization


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # We'll use dpM to store two contributions that tells us which rows of X we
    # should to include in the calculation of dW = X.T.dot(dpM)
    dpM = np.zeros((X.shape[0], W.shape[1]))   # N x C zeros

    # first contributoin (all active margins for others)
    pMactive = np.where(M>0, 1, 0)
    dpM += pMactive

    # second contributoin subtract fro self self sum of others active
    dpM[mask] = -1*np.sum(pMactive, axis=1)

    # gadiaent
    dW = 1.0/num_train * X.T.dot(dpM)   +   2*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
