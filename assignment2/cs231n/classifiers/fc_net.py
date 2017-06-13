from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


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

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
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
        D = input_dim
        M = hidden_dim
        C = num_classes
        
        # The network transforms an X of [N x D] to an output of [N x C], so:
        # [N x D] dot [D x M] ==> [N x M]
        # [N x M] dot [M x C] ==> [N x C]
        W1, b1 = weight_scale * np.random.randn(D, M), np.zeros(M)
        W2, b2 = weight_scale * np.random.randn(M, C), np.zeros(C)
        
        self.params['W1'] = W1
        self.params['b1'] = b1
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
        
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        
        N = X.shape[0]
        X_vec = X.reshape(N, -1)
        
        # zi is the output of the ith layer as per notation convention
        # ai is the activation of the ith layer
        z1, z1_cache = affine_forward(X_vec, W1, b1)
        a1, a1_cache = relu_forward(z1)
        z2, z2_cache = affine_forward(a1, W2, b2)
        
        scores = z2 # scores is just an alias of the output of our last layer
        
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
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # L is data loss
        L, dL = softmax_loss(scores, y)
        # Reg loss is only a function of W and is not involved in the backward pass
        reg_loss = 0.5 * self.reg * ( np.sum(W1*W1) + np.sum(W2*W2) )
        
        loss = L + reg_loss
        
        # Backprop. Layerwise-defined backwards functions does not account for regularization
        #   gradients (because those functions are not aware of the quantity), so we need to 
        #   add those in separately here for each W
        da1, dW2, db2 = affine_backward(dL, z2_cache)
        dW2 += self.reg * W2
        dz1 = relu_backward(da1, a1_cache)
        dX_vec, dW1, db1 = affine_backward(dz1, z1_cache)
        dW1 += self.reg * W1
        
        # Finally, reshape dX into its original dimensions
        dX = dX_vec.reshape(X.shape)
        
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
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
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        
        # We store layer outputs and caches in the same format as params
        self.outs = {}
        self.caches = {}
        
        D = input_dim
        C = num_classes
        
        # The layers successively transform inputs from:
        # [N x D] l1 ==> [N x M1] l2 ==> [N x M2] l3 ==> ... ==> [N x C]
        # Therefore, layer sizes go like this:
        # [D x M1] W1 ==> [M1 x M2] W2 ==> [M2 x M3] W3 ==> ...
        # Biases go from vectors of size M1 in l1, to M2 in l2, ... C
        dims = []
        dims.append(D)
        dims.extend(hidden_dims)
        dims.append(C)
        
        for i in range(self.num_layers):
            self.params['W'+str(i+1)] = weight_scale * np.random.randn(dims[i], dims[i+1])
            self.params['b'+str(i+1)] = np.zeros(dims[i+1])
            
        if self.use_batchnorm:
            for i in range(self.num_layers - 1):
                # Initialize gamma to 1 and beta to 0 as per instructions
                self.params['gamma'+str(i+1)] = np.ones(dims[i+1])
                self.params['beta'+str(i+1)] = np.zeros(dims[i+1])
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

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

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
    
        N = X.shape[0]
        X_vec = X.reshape(N, -1)
        
        for i in range(self.num_layers):
            
            # Input vector to layer is either X_vec for the first layer, or it would be
            #   the output of the activation of layer before it
            layer_input = X_vec if (i == 0) else self.outs['a'+str(i)]
            
            # If we have dropout, then it goes before the affine layer
            if self.use_dropout:
                dropi, dropi_cache = dropout_forward(layer_input, self.dropout_param)
                self.outs['dropout'+str(i+1)] = dropi
                self.caches['dropout'+str(i+1)] = dropi_cache
            
            # Forward through affine
            affine_input = layer_input if not self.use_dropout else dropi
            
            Wi = self.params['W'+str(i+1)]
            bi = self.params['b'+str(i+1)]
            zi, zi_cache = affine_forward(affine_input, Wi, bi)
            self.outs['z'+str(i+1)] = zi
            self.caches['z'+str(i+1)] = zi_cache
            
            # The last layer doesn't go through an activation or batchnorm,
            #   so break here if this condition is true
            if i == self.num_layers - 1:
                scores = zi
                break
                
            # If we have batchnorm, then the layer goes between affine and activation
            if self.use_batchnorm:
                gammai = self.params['gamma'+str(i+1)]
                betai = self.params['beta'+str(i+1)]
                bn_parami = self.bn_params[i] # Pass self.bn_params[0] to 1st layer as per instructions
                bni, bni_cache = batchnorm_forward(zi, gammai, betai, bn_parami)
                self.outs['bn'+str(i+1)] = bni
                self.caches['bn'+str(i+1)] = bni_cache
                
            # Forward through activation
            # Depending on whether we have batchnorm, the activation input differs
            activation_input = zi if not self.use_batchnorm else bni
            
            ai, ai_cache = relu_forward(activation_input)
            self.outs['a'+str(i+1)] = ai
            self.caches['a'+str(i+1)] = ai_cache
        
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
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # Calculate loss
        L, dL = softmax_loss(scores, y)
        reg_loss = 0
        for i in range(self.num_layers):
            _W = self.params['W'+str(i+1)]
            reg_loss += np.sum(_W*_W)
        reg_loss = 0.5 * self.reg * reg_loss
        loss = L + reg_loss
        
        # Backprop. Layer indexing starts from 1 by convention
        for i in reversed(range(self.num_layers)):
            
            # If this is the top layer, we don't have an activation, and
            #   d wrt output is just d wrt loss
            if i == self.num_layers - 1:
                dlayer_output = dL
                
            # If this isn't the top layer, we have an activation to backprop through.
            #   d wrt activation's output is dai.
            else:
                ai_cache = self.caches['a'+str(i+1)]
                dai = relu_backward(dzi, ai_cache)
                dlayer_output = dai
            
                # If we have batchnorm then this goes between the activation and affine layers
                # Top layer does not go through batchnorm
                if self.use_batchnorm:
                    bni = self.outs['bn'+str(i+1)]
                    bni_cache = self.caches['bn'+str(i+1)]
                    dbni, dgammai, dbetai = batchnorm_backward(dlayer_output, bni_cache)
                    dlayer_output = dbni
                    grads['gamma'+str(i+1)] = dgammai
                    grads['beta'+str(i+1)] = dbetai
            
            # Backprop through affine
            Wi = self.params['W'+str(i+1)]
            zi_cache = self.caches['z'+str(i+1)]
            dzi, dWi, dbi = affine_backward(dlayer_output, zi_cache)
            dWi += self.reg * Wi
            
            grads['W'+str(i+1)] = dWi
            grads['b'+str(i+1)] = dbi
            
            # If we have dropout, backprop through dropout
            if self.use_dropout:
                dropi_cache = self.caches['dropout'+str(i+1)]
                ddropi = dropout_backward(dzi, dropi_cache)
                dzi = ddropi
            
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
