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
  train_num = X.shape[0]
  class_num = W.shape[1]
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(train_num):
    score = X[i].dot(W)
    score -= max(score)
    loss = loss + np.log(np.sum(np.exp(score))) - score[y[i]]
    for j in range(class_num):
        dW[:,j] += np.exp(score[j])*X[i]/np.sum(np.exp(score))
    dW[:, y[i]] -= X[i]
  loss /=train_num
  dW /=train_num
  loss +=reg*np.sum(W**2)
  dW +=2*reg*W

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
  train_num = X.shape[0]
  class_num = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = X.dot(W)
  score -=np.max(score,axis=1).reshape(train_num,1)
  correct_class = np.exp(score[np.arange(train_num),y])
  sum_per_train = np.sum(np.exp(score),axis=1)
  loss = np.sum(-np.log(correct_class/sum_per_train))
  dS = np.zeros_like(score)
  dS = np.exp(score)/sum_per_train.reshape(train_num,1)
  dS[np.arange(train_num),y] -= 1
  
  dW = X.T.dot(dS)
  
  loss /= train_num
  loss += reg*np.sum(W**2)
  dW /=train_num
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

