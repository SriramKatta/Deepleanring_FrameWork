from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.Sigmoid import Sigmoid
from Layers.TanH import TanH

import numpy as np  

class RNN(BaseLayer):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.trainable = True
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.fc1 = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
    self.fc2 = FullyConnected(self.hidden_size, self.output_size)
    self.initialize()
    self.hidden_state = np.zeros((1, self.hidden_size))
    self.grad_hidden_states = np.zeros((1, self.hidden_size))
    self.memval = False
    self.optimizerval = None
    self.tanh = TanH()
    self.sig = Sigmoid()
    self.grad_weights = None
  
  def initialize(self, weights_initializer=None, bias_initializer=None):
    if weights_initializer is None or bias_initializer is None:
      return
    self.fc1.initialize(weights_initializer, bias_initializer)
    self.fc2.initialize(weights_initializer, bias_initializer)

  def forward(self, input_tensor):
    pass

  def backward(self, error_tensor):
    pass

  def calculate_regularization_loss(self):
    pass

  @property
  def memorize(self):
    return self.memval
  @memorize.setter
  def memorize(self, newval):
    self.memval = newval

  @property
  def optimizer(self):
    return self.optimizerval
  @optimizer.setter
  def optimizer(self, newval):
    self.optimizerval = newval