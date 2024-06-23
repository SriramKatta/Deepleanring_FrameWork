import numpy as np
import copy
from Layers import Base
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid


class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.optimizerval = None
        self.tanhlay = TanH()
        self.siglayer = Sigmoid()
        self.fc_hidden = FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_out = FullyConnected(hidden_size, output_size)
        self.h_t = None
        self.h_t1 = None
        self.y_t = None
        self.memval = False

    def forward(self, input_tensor):
        self.input = input_tensor
        self.batchsize = input_tensor.shape[0]
        self.output = np.zeros((self.batchsize, self.output_size))

        if self.h_t1 is None:
            self.h_t1 = np.zeros(self.hidden_size)
        
        for time in  range(self.batchsize):
            xt = input_tensor[time]
            hxt = np.concatenate((xt, self.h_t1), axis=None).reshape(1,-1)
            self.h_t1 = self.tanhlay.forward(self.fc_hidden.forward(hxt))
            self.y_t = self.fc_out.forward(self.h_t1)
            self.output[time] = self.siglayer.forward(self.y_t)

        return self.output
    
    def backward(self, error_tensor):
        error_tensor_prev = np.zeros_like(self.input)

        for time in range(self.batchsize):
            pass

        return error_tensor_prev

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_out.initialize(weights_initializer, bias_initializer)

    @property
    def optimizer(self):
        return self.optimizerval

    @optimizer.setter
    def optimizer(self, newval):
        self.optimizerval = copy.deepcopy(newval)

    @property
    def memorize(self):
        return self.memval
    
    @memorize.setter
    def memorize(self, newval):
        self.memval = newval
    
    @property
    def weights(self):
        return self.fc_hidden.weights
    
    @weights.setter
    def weights(self, newval):
        self.fc_hidden.weights = newval

    @property
    def gradient_weights(self):
        return self.fc_hidden.weights
    
    @gradient_weights.setter
    def gradient_weights(self, newval):
        self.fc_hidden.gradient_weights = newval
        self.fc_out.gradient_weights = newval

