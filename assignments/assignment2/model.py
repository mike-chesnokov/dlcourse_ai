import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        
        self.hidden_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.non_linearity = ReLULayer()
        self.output_layer = FullyConnectedLayer(hidden_layer_size, n_output)


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        params_ = self.params()
        for param in params_:
            params_[param].clear_grad() 
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        temp_res = self.hidden_layer.forward(X)
        temp_res = self.non_linearity.forward(temp_res)
        temp_res = self.output_layer.forward(temp_res)
        
        loss, dpred = softmax_with_cross_entropy(temp_res, y)
        
        temp_grad = self.output_layer.backward(dpred)
        temp_grad = self.non_linearity.backward(temp_grad)
        temp_grad = self.hidden_layer.backward(temp_grad)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        params_ = self.params()
        for param in params_:
            loss_l2, grad_l2 = l2_regularization(params_[param].value, self.reg)
            loss += loss_l2
            params_[param].grad += grad_l2

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        
        temp_res = self.hidden_layer.forward(X)
        temp_res = self.non_linearity.forward(temp_res)
        pred = self.output_layer.forward(temp_res)
        
        pred = np.argmax(softmax(pred), axis=1)
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        result['W_h'] = self.hidden_layer.W
        result['B_h'] = self.hidden_layer.B
        result['W_o'] = self.output_layer.W
        result['B_o'] = self.output_layer.B        
                
        return result
