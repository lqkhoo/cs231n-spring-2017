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
  
  
  def _softmax(v):
    """Given a vector v, returns its softmax"""
    s = v - np.max(v) # shift
    s = np.exp(s) # exponentiate
    return np.divide(s, np.sum(s)) # softmax
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  for i in xrange(num_train): # For every point
        
    score_i = X[i].dot(W)
    S = _softmax(score_i) # [C x 1] vector of e^s_j / sum(e^s_j) forall j
    loss += -np.log(S[y[i]]) # y[i] is the label
    
    for j in xrange(num_classes):
      dW[:, j] += (S[j] - (y[i]==j).astype(int)) * X[i]
        

  loss /= num_train
  dW /= num_train
  
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  
  # with reference to:
  #   http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  #   http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture3.pdf
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
    
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  
  # Compute softmax matrix
  S = (scores.T - np.max(scores, axis=1).T).T # Transpose for broadcast
  S = np.exp(S)
  S = (S.T / np.sum(S, axis=1).T).T # Transpose for broadcast

  # For the loss, we take the entries in the softmax matrix corresponding 
  #   to the correct labels and sum their -logs
  softmax_losses = -np.log(S[np.arange(num_train), y])
  loss = np.sum(softmax_losses)
  
  # For dW, we need to first construct a matrix M that comprises of vectorized
  #   operations between S and the Kronecker delta
  kd_matrix = np.zeros(S.shape)
  kd_matrix[np.arange(num_train), y] = 1
  M = S - kd_matrix
  
  dW = X.T.dot(M)

  loss /= num_train
  dW /= num_train
  
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
    
  # with reference to:
  #   http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

