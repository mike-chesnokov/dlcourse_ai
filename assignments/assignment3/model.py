import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization,
    softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        # padding=0, stride=1
        self.padding = 0
        self.stride = 1
        self.filter_size = 3
        self.pool_size = 4
        
        self.conv1_out_size = int((input_shape[0] - self.filter_size + 2*self.padding)/self.stride + 1)
        self.max_pool1_out_size = int((self.conv1_out_size - self.pool_size)/self.stride + 1)
        self.conv2_out_size = int((self.max_pool1_out_size - self.filter_size + 2*self.padding)/self.stride + 1)
        self.max_pool2_out_size = int((self.conv2_out_size - self.pool_size)/self.stride + 1)
        
        # padding=0, stride=1
        self.sequence = [
            ConvolutionalLayer(input_shape[2], conv1_channels, self.filter_size, self.padding),
            ReLULayer(),
            MaxPoolingLayer(self.pool_size, self.stride),
            ConvolutionalLayer(conv1_channels, conv2_channels, self.filter_size, self.padding),
            ReLULayer(), 
            MaxPoolingLayer(self.pool_size, self.stride),
            Flattener(),
            FullyConnectedLayer(self.max_pool2_out_size  * self.max_pool2_out_size * conv2_channels, n_output_classes)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        params_ = self.params()
        for param in params_:
            params_[param].clear_grad() 
        
        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        for ind, layer in enumerate(self.sequence):
            X = layer.forward(X)
            #print('Layer {} forward done...'.format(str(ind)))
           
        loss, dpred = softmax_with_cross_entropy(X, y)
        
        for ind, layer in enumerate(self.sequence[::-1]):
            dpred = layer.backward(dpred)
            #print('Layer {} backward done...'.format(str(ind)))
            
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        for ind, layer in enumerate(self.sequence):
            X = layer.forward(X)
        
        pred = np.argmax(softmax(X), axis=1)
        
        return pred
        
    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        for ind, layer in enumerate(self.sequence):
            params = layer.params()
            
            if len(params) > 0:
                for param in params:
                    result[param +'_'+ str(ind)] = params[param]

        return result
