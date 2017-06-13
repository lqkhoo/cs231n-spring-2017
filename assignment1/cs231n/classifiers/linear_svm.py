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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
    
  # Iterate through each point, summing as we go, and then divide by n afterwards
  # Issues with numerical precision using this approach (?)
  for i in xrange(num_train):
    
    scores = X[i].dot(W) # C-dimensional vector
    correct_class_score = scores[y[i]]
    
    # Now compute margins across all classes for this point X[i]
    num_positive_margin = 0
    for j in xrange(num_classes):  
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        num_positive_margin += 1
        dW[:, j] += X[i]
    dW[:, y[i]] -= num_positive_margin * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
    
  # With reference to:
  # http://cs231n.github.io/optimization-1/#computing-the-gradient-analytically-with-calculus

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  
  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  scores = X.dot(W) # [N x C] # [500 x 10]

  # Produce a column-wise one-hot boolean (binary) integer matrix from y
  true_class_mask = np.zeros(scores.shape)
  true_class_mask[np.arange(num_train), y] = 1
    
  # To get the vector of true class scores, we simply multiply the scores with
  #   the mask in order to get another [N x C] matrix, and sum along columns
  #   to get an N-dimensional vector
  true_class_scores = np.sum(scores * true_class_mask, axis=1)
  
  # Now we can broadcast these true class scores to compute the matrix of margins,
  #   which is also [N x C]. We hardcode delta to 1
  #   Broadcasting requires the same trailing dimensions,
  #   so we transpose and transpose back
  margins = (scores.T - true_class_scores.T).T + 1
  
  # Now we need to zero the margins corresponding to the true class label
  #   we can do this by an elementwise multiply with the (boolean) negative 
  #   of the true class matrix
  margin_mask = -true_class_mask + 1
  margins = margins * margin_mask
  
  # Now we zero the margins which are less than 0 to represent the max(0, .) operation
  margin_mask2 = (margins > 0).astype(int)
  margins = margins * margin_mask2

  # And finally the loss is simply the sum of all remaining margins, divided by N
  loss = np.sum(margins) / num_train
    
  # Account for regularization
  loss += reg * np.sum(W * W)
  

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  
  # Compute wrt w for j != y[i]
  dW1 = X.T.dot(margin_mask2) # View as matrix product == sum of vector outer products

  # Compute wrt j == y[i]
  margins_boolean = (margins > 0).astype(int)
  num_positive_margins = np.sum(margins_boolean, axis=1).reshape(num_train, 1)
  dW2 = X.T.dot(-true_class_mask * num_positive_margins)
  dW = (dW1.T + dW2.T).T / num_train
  
  # Account for regularization
  dW += 2 * reg * W
  
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


def svm_loss_naive_original(W, X, y, reg):
    
  """ This is the original unmodified function as given in the assignment"""

  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  loss += reg * np.sum(W * W) # Add regularization to the loss.
  return loss, dW

