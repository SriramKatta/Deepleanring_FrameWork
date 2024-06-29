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
        self.fc_hidden = FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_out = FullyConnected(hidden_size, output_size)
        self.h_t = None
        self.h_t_prevbatch = None
        self.y_t = None
        self.memval = False

    def forward(self, input_tensor):
        self.input = input_tensor
        self.batch_size = input_tensor.shape[0]
        output = np.zeros((self.batch_size, self.output_size))
        if self.memval == True:
            if self.h_t is None:
                self.h_t = np.zeros((self.batch_size+1, self.hidden_size))
            else :
                self.h_t[0] = self.h_t_prevbatch
        else:
            self.h_t = np.zeros((self.batch_size, self.output_size))

        for time in  range(self.batch_size):
            xt = input_tensor[time][np.newaxis, :]
            hidd_t1 = self.h_t[time][np.newaxis, :]
            x_tilde = np.hstack((xt, hidd_t1))
            
            ut = self.fc_hidden.forward(x_tilde)
            self.h_t[time + 1] = self.tanhlay.forward(ut)
            ot = self.fc_out.forward(self.h_t[time + 1].reshape(1,-1))
            output[time] = self.siglayer.forward(ot)

        self.h_t_prevbatch = self.h_t[-1]
        return output
    
    def backward(self, error_tensor):
        error_tensor_prev = np.zeros_like(self.input)
        hidden_error = np.zeros((1, self.hidden_size))

        for revtime in reversed(range(self.batch_size)):
            grad_o_t = self.siglayer.backward(error_tensor[revtime + 1].reshape((1,-1)))
            self.fc_out.input_tensor_val = self.h_t[revtime + 1]
            self.fc_out.backward(grad_o_t.reshape(1,-1))


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

