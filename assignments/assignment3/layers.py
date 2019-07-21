import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength * np.sum(W**2)
    grad = 2 * reg_strength * W

    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    preds = predictions.copy()
    
    if preds.ndim > 1:
        maxs = np.max(preds, axis=1)[:, np.newaxis]
        preds -= maxs
        exps = np.exp(preds)
        probs = exps/np.sum(exps, axis=1)[:, np.newaxis]
        
    else:
        preds -= np.max(preds)
        exps = np.exp(preds)
        probs = exps/np.sum(exps)
        
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if probs.ndim > 1:
        loss = -1 * np.sum(np.log(probs[np.arange(probs.shape[0]), target_index.flatten()]))
    else:
        loss = -1 * np.log(probs[target_index])
        
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    batch_size = probs.shape[0]
    
    if predictions.ndim > 1:
        dprediction = probs.copy()
        dprediction[np.arange(batch_size), target_index.flatten()] -= 1
        
        return loss/batch_size, dprediction/batch_size
    else:
        dprediction = probs.copy()
        dprediction[target_index] -= 1
    
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.clear_grad()
    
    def clear_grad(self):
        self.grad = np.zeros_like(self.value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.grad = (X > 0).astype(float)
        
        return np.maximum(0, X)

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_result = d_out * self.grad
        
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        
        return X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        # TODO copy from the previous assignment
        
        d_input = d_out.dot(self.W.value.T)
        
        self.W.grad += self.X.T.dot(d_out)
        self.B.grad += d_out.sum(axis=0)

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.padding = padding
        self.X = None
       

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height - self.filter_size + 2*self.padding + 1
        out_width = width - self.filter_size + 2*self.padding + 1
        
        # add padding
        if self.padding > 0:
            X_padded = np.zeros((batch_size, height + 2*self.padding, 
                                 width + 2*self.padding, channels))
            X_padded[:, self.padding : height + self.padding, 
                        self.padding :  width + self.padding, :] = X.copy()
            X = X_padded.copy()
            
        self.X = X.copy()
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                X_temp = X[:, x : x+self.filter_size, y: y+self.filter_size, :]\
                            .reshape(batch_size, self.filter_size * self.filter_size * channels)
                W_temp = self.W.value.reshape(self.filter_size * self.filter_size * channels, self.out_channels)
                result[:, x, y, :] = np.dot(X_temp, W_temp) + self.B.value

        return result

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        d_input = np.zeros(self.X.shape)
        
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                
                # gradients over input
                d_out_temp = d_out[:, x, y, :].reshape(batch_size, out_channels)
                W_temp = self.W.value.reshape(self.filter_size * self.filter_size * channels, self.out_channels)
                #print(d_out_temp.shape, W_temp.T.shape)
                res_temp = np.dot(d_out_temp, W_temp.T)\
                                .reshape(batch_size, self.filter_size, self.filter_size, channels)
                d_input[:, x : x+self.filter_size, y : y+self.filter_size, :] += res_temp
                
                # gradient over weights and biases
                X_temp = self.X[:, x : x+self.filter_size, y: y+self.filter_size, :]\
                                .reshape(batch_size, self.filter_size * self.filter_size * channels)
                self.W.grad += np.dot(X_temp.T, d_out_temp)\
                                .reshape(self.filter_size, self.filter_size, channels, self.out_channels)
                self.B.grad += d_out_temp.sum(axis=0)

        if self.padding > 0:
            d_input = d_input[:, self.padding : height - self.padding, 
                                 self.padding :  width - self.padding, :]
        return d_input
        
    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.max_inds = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        self.X = X.copy()
        
        out_height = int((height - self.pool_size)/self.stride + 1)
        out_width = int((width - self.pool_size)/self.stride + 1)
        
        result = np.zeros((batch_size, out_height, out_width, channels))
        
        self.max_inds = np.zeros((batch_size, out_height, out_width, channels))
        
        for y in range(out_height):
            for x in range(out_width):        
                X_temp = X[:, x : x+self.pool_size, y: y+self.pool_size, :]\
                            .reshape(batch_size, self.pool_size * self.pool_size, channels)
                cur_max = np.max(X_temp, axis=1) # max along heght or width not channel, not batch
                # save indexies and results
                self.max_inds[:, x, y, :] = np.argmax(X_temp, axis=1)
                result[:, x, y, :] = cur_max
        
        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        d_input = np.zeros(self.X.shape)#.reshape(batch_size, height * width, channels)
        
        for y in range(out_height):
            for x in range(out_width):        
                # flattern output and indices
                d_out_temp = d_out[:, x, y, :].reshape(-1)
                max_inds_temp = self.max_inds[:, x, y, :].reshape(-1).astype(int)
                
                # prepare zeros input
                d_input_temp = np.zeros((batch_size, self.pool_size * self.pool_size, channels))
                
                # prepare indices for multi-indexing
                batch_inds = np.repeat(np.arange(batch_size), channels)
                channel_inds = np.array([np.arange(channels) for x in range(batch_size)]).reshape(-1)
                #print(len(batch_inds), len(max_inds_temp), len(channel_inds))
                assert batch_inds.shape == channel_inds.shape, 'b={}, ch={}'.format(batch_inds.shape, channel_inds.shape)
                assert batch_inds.shape == max_inds_temp.shape, 'b={}, inds={}'.format(batch_inds.shape, max_inds_temp.shape)
                
                # fill 3d tensor with values
                d_input_temp[batch_inds, max_inds_temp, channel_inds] = d_out_temp
                # reshape 3d tensor to 4d
                d_input[:, x : x+self.pool_size, 
                           y : y+self.pool_size, :] += \
                    d_input_temp.reshape(batch_size, self.pool_size, self.pool_size, channels)
                
        return d_input        
                
    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        batch_size, height, width, channels = self.X_shape
        
        return d_out.reshape(batch_size, height, width, channels)

    def params(self):
        # No params!
        return {}
