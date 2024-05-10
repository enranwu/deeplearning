import numpy as np

from deeplearning.layers import *
from deeplearning.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        pass
        # first layer parameters
        W1 = np.random.normal(0, weight_scale, size=(input_dim, hidden_dim))
        b1 = np.zeros(hidden_dim)
        self.params['W1'] = W1
        self.params['b1'] = b1

        # second layer parameters
        W2 = np.random.normal(0, weight_scale, size=(hidden_dim, num_classes))
        b2 = np.zeros(num_classes)
        self.params['W2'] = W2
        self.params['b2'] = b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # first layer
        out1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        # second layer
        out2, cache2 = affine_forward(out1, self.params['W2'], self.params['b2'])
        scores = out2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization on the weights,    #
        # but not the biases.                                                      #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        loss, dscores = softmax_loss(scores, y)

        dx2, grads['W2'], grads['b2'] = affine_backward(dscores, cache2)
        dx1, grads['W1'], grads['b1'] = affine_relu_backward(dx2, cache1)

        # L2 regularization i.e. 1/2*constant*(||w1||^2 + ||w2||^2)
        loss += 1/2*self.reg*(np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        grads['W2'] += self.reg*self.params['W2']
        grads['W1'] += self.reg*self.params['W1']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. For a network with L layers,
    the architecture will be

    {affine - relu} x (L - 1) - affine - softmax

    where the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        """

        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        ############################################################################
        pass
        # full dimensions i.e. [input_dim, ---hidden_dims---, num_classes]
        total_dims = [input_dim] + hidden_dims + [num_classes]  
        for i in np.arange(1, self.num_layers+1):
          W_i_name = 'W' + str(i)
          b_i_name = 'b' + str(i)
          # dimension matching with current layer and next layer
          W_i = np.random.normal(0, weight_scale, size=(total_dims[i-1], total_dims[i]))
          b_i = np.zeros(total_dims[i])
          self.params[W_i_name] = W_i
          self.params[b_i_name] = b_i
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #                                                       #
        ############################################################################
        pass
        # initialize OUT variable and caches
        out = X
        self.caches = {}

        for i in np.arange(1, self.num_layers + 1):
          # W, b, cache names
          W_i_name = 'W' + str(i)
          b_i_name = 'b' + str(i)
          cache_i_name = 'cache' + str(i)

          # {affine - relu} x (L - 1) - affine - softmax
          # 1, 2, ...                  , L
          # perform affine in the final layer, otherwise do affine_relu
          if i == self.num_layers:
            out, cache_i = affine_forward(out, self.params[W_i_name], self.params[b_i_name])
          else:
            out, cache_i = affine_relu_forward(out, self.params[W_i_name], self.params[b_i_name])

          # store cache in every layer
          self.caches[cache_i_name] = cache_i
        
        # store final scores
        scores = out

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization on the         #
        # weights, but not the biases.                                             #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        # initialize loss and dscores with softmax_loss
        loss, dscores = softmax_loss(scores, y)

        # softmax - affine - {affine - relu} x (L - 1)
        # --------- L, L - 1, L - 2, ..., 1
        # perform affine in the last layer and affine_relu else until reaching first layer
        for i in np.arange(self.num_layers, 0, -1):
          # W, b, cache names
          W_i_name = 'W' + str(i)
          b_i_name = 'b' + str(i)
          cache_i_name = 'cache' + str(i)
          cache_i = self.caches[cache_i_name]

          if i == self.num_layers:
            dscores, grads[W_i_name], grads[b_i_name] = affine_backward(dscores, cache_i)
          else:
            dscores, grads[W_i_name], grads[b_i_name] = affine_relu_backward(dscores, cache_i)

          # L2 regularization i.e. 1/2*constant*(||w1||^2 + ||w2||^2)
          loss += 1/2*self.reg*(np.sum(self.params[W_i_name]**2))
          grads[W_i_name] += self.reg*self.params[W_i_name]

          
          

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
