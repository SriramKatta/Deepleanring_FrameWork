import numpy as np
import copy 
from Layers import Base
from Layers import Sigmoid
from Layers import TanH
from Layers import FullyConnected as FC


class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.memorize_val = False
        self.optimizer_val = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_state = [np.zeros(self.hidden_size)]
        
        self.fc_hidden = FC.FullyConnected(input_size+hidden_size, hidden_size)
        self.tanh_lay = TanH.TanH()
        self.sig_lay = Sigmoid.Sigmoid()
        self.fc_out = FC.FullyConnected(hidden_size, output_size)
        self.gradient_weights_val = None
        self.gradient_weights_hy_val = None
        self.weights = self.fc_hidden.weights
        self.weights_hy = None

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_out.initialize(weights_initializer, bias_initializer)
        self.weights = self.fc_hidden.weights
        self.weights_hy = self.fc_out.weights

    def forward(self, input_tensor):
        if self.memorize == False:
            self.hidden_state = [np.zeros((1, self.hidden_size))]

        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]
        self.output = np.zeros((self.batch_size, self.output_size))
        #self.x_tilde_mem = []
        for time, intensor in enumerate(input_tensor):
            x_t = intensor.reshape(1, -1)
            h_t1 = self.hidden_state[-1].reshape(1, -1)
            x_tilde = np.hstack([x_t, h_t1])
            #self.x_tilde_mem.append(x_tilde)
            u_t = self.fc_hidden.forward(x_tilde)
            self.hidden_state.append(self.tanh_lay.forward(u_t))
            o = self.fc_out.forward(self.hidden_state[-1])
            self.output[time] = self.sig_lay.forward(o)

        return self.output

    def backward(self, error_tensor):

        self.gradient_weights_val = np.zeros_like(self.fc_hidden.weights)
        self.gradient_weights_hy_val = np.zeros_like(self.fc_out.weights)
        output_error = np.zeros((self.batch_size, self.input_size))
        error_h = np.zeros((1, self.hidden_size))

        # bp through time
        for revtime in reversed(range(error_tensor.shape[0])):
            #forward to set properactivation
            x_t = self.input_tensor[revtime,:].reshape(1, -1)
            h_t1 = self.hidden_state[revtime].reshape(1, -1)
            x_tilde = np.hstack([x_t, h_t1])
            self.sig_lay.forward(self.fc_out.forward(self.tanh_lay.forward(self.fc_hidden.forward(x_tilde))))
            #setting proper activations for all layers
            #self.fc_hidden.activation = self.x_tilde_mem[revtime]
            #self.tanh_lay.activation = self.hidden_state[revtime]
            #self.fc_out.activation = self.hidden_state[revtime]
            #self.sig_lay.activation = self.output[revtime]
            # backward
            grad = self.sig_lay.backward(error_tensor[revtime, :])
            grad = self.fc_out.backward(grad) + error_h
            self.gradient_weights_hy_val += self.fc_out.gradient_weights
            grad = self.tanh_lay.backward(grad)
            grad = self.fc_hidden.backward(grad)
            self.gradient_weights_val += self.fc_hidden.gradient_weights
            output_error[revtime, :] = grad[:, :self.input_size]
            error_h = grad[:, self.input_size:]

        if self.optimizer:
            self.fc_hidden.weights = self.optimizer.calculate_update(
                self.fc_hidden.weights, self.gradient_weights_val)
            self.fc_out.weights = self.optimizer.calculate_update(
                self.fc_out.weights, self.gradient_weights_hy_val)

        self.weights = self.fc_hidden.weights
        self.weights_hy = self.fc_out.weights

        return output_error

    @property
    def memorize(self):
        return self.memorize_val

    @memorize.setter
    def memorize(self, memorize):
        self.memorize_val = memorize

    @property
    def weights(self):
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, weights):
        self.fc_hidden.weights = weights

    @property
    def gradient_weights(self):
        return self.gradient_weights_val

    @gradient_weights.setter
    def gradient_weights(self, new_weights):
        self.fc_hidden._gradient_weights = new_weights

    @property
    def optimizer(self):
        return self.optimizer_val

    @optimizer.setter
    def optimizer(self, optimizer):
        self.optimizer_val = copy.deepcopy(optimizer)

