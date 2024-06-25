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
        self.batchsize = input_tensor.shape[0]
        self.output = np.zeros((self.batchsize, self.output_size))
        if self.memval == True:
            if self.h_t is None:
                self.h_t = np.zeros((self.batchsize+1, self.hidden_size))
            else :
                self.h_t[0] = self.h_t_prevbatch
        else:
            self.h_t = np.zeros_like(self.h_t)

        self.x_tilde_mem = []
        self.tanh_mem = []
        self.sig_mem = []
        self.h_t_mem = []
    
        for time in  range(self.batchsize):
            xt = input_tensor[time].reshape((1,-1))
            hidd_t1 = self.h_t[time].reshape((1,-1))
            x_tilde = np.hstack((hidd_t1, xt))
            
            ut = self.fc_hidden.forward(x_tilde)
            self.x_tilde_mem.append(self.fc_hidden.input_tensor)
            
            self.tanh_mem.append(ut)
            ht = self.tanhlay.forward(ut)
            
            self.h_t[time + 1] = ht
            ot = self.fc_out.forward(ht)
            self.h_t_mem.append(self.fc_out.input_tensor)
            self.sig_mem.append(ot)
            self.output[time] = self.siglayer.forward(ot)

        self.h_t_prevbatch = self.h_t[-1]
        return self.output
    
    def backward(self, error_tensor):
        error_tensor_prev = np.zeros_like(self.input)

        for revtime in reversed(range(self.batchsize)):
            self.siglayer.activation = self.sig_mem[revtime]
            self.fc_hidden.input_tensor = self.x_tilde_mem[revtime]
            self.tanhlay.activation = self.tanh_mem[revtime]
            self.fc_out.input_tensor = self.h_t_mem[revtime]

            curr_e_tensor = error_tensor[revtime].reshape((1,-1))
            curr_e_tensor = self.siglayer.backward(curr_e_tensor)
            curr_e_tensor = self.fc_hidden.backward(curr_e_tensor)
            curr_e_tensor = self.tanhlay.backward(curr_e_tensor)
            error_tensor_prev[revtime] = self.fc_out.backward(curr_e_tensor)

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

