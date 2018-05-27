import numpy as np
from random import shuffle

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
  # compute loss
  wx = X.dot(W)
  p_unnorm = np.exp(wx)
  p_sum = np.sum(p_unnorm,1)[:,None]
  p_sum_inv = 1/p_sum
  p = p_unnorm * p_sum_inv
  yp = np.choose(y, p.T)
  p_log = np.log(yp)
  loss = -np.mean(p_log)
  reg_loss = reg*np.sum(W**2)
  loss += reg_loss

  # compute local gradients
  dloss = 1
  dp_log = np.zeros_like(p_log)
  dp_log[:] = -1/len(dp_log)
  dyp = 1/yp
  dp = np.zeros_like(p)
  dp[np.arange(len(dp)), y]=1
  dp_sum_inv = p_unnorm
  dp_unnorm = p_sum_inv
  dp_sum = -p_sum**-2
  dp_unnorm2 = np.ones_like(p_unnorm)
  dwx = np.exp(wx)
  dw = X
  dw_reg = 2*reg*W

  # compute gradients of loss wrt W using chain rule
  dLp_log = dloss * dp_log
  dLyp = dLp_log * dyp
  dLp = dLyp[:,None] * dp
  dLp_sum_inv = (dLp * dp_sum_inv).sum(1)
  dLp_unnorm = dLp * dp_unnorm
  dLp_sum = dLp_sum_inv[:,None] * dp_sum
  dLp_unnorm2 = dLp_sum * dp_unnorm2
  dLp_unnorm += dLp_unnorm2
  dLwx = dLp_unnorm * dwx
  dW = np.dot(dw.T, dLwx)
  dW += dw_reg
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
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return softmax_loss_naive(W, X, y, reg)

