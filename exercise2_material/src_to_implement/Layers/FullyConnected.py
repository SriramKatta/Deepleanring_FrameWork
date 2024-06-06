from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
  def __init__(self,input_size, output_size):
    super().__init__()
    self.trainable = True
    self.input_size = input_size
    self.output_size = output_size
    self.weights = np.random.rand(input_size + 1 , output_size)
    self.input_tensor = None
    self.optimizerval = None
    self.gradient_weightsval = None

  def forward(self, input_tensor):
    batchsize, _ = np.shape(input_tensor)
    self.input_tensor = np.c_[input_tensor, np.ones(batchsize)]
    return self.input_tensor @ self.weights    

  def backward(self, error_tensor):
    self.gradient_weightsval = self.input_tensor.T @ error_tensor
    if(self.optimizerval != None):
      self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
    return error_tensor @ self.weights[:-1,:].T

  def calculate_update(self, weight_tensor, gradient_tensor):
    if(self.optimizer == None):
      return
    self.optimizer.calculate_update(weight_tensor, gradient_tensor)
  
  @property
  def optimizer(self):
    return self.optimizerval
  
  @optimizer.setter
  def optimizer(self,newval):
    self.optimizerval = newval

  @property
  def gradient_weights(self):
    return self.gradient_weightsval
  
  @gradient_weights.setter
  def gradient_weights(self,newval):
    self.gradient_weightsval = newval

  def initialize(self, weights_initializer, bias_initializer):
    input_dim, output_dim = self.weights.shape
    self.weights[:self.input_size, : ] = weights_initializer.initialize(self.weights[:-1,:].shape, input_dim - 1, output_dim)
    self.weights[-1, :] = bias_initializer.initialize(self.weights[-1,:].shape, input_dim - 1, output_dim)
  