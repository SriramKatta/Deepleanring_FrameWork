from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients

import numpy as np

class BatchNormalization(BaseLayer): 
  def __init__(self, channels):
    super().__init__()
    self.trainable = True
    self.channels = channels
    self.weights = np.ones(self.channels, dtype=float)
    self.bias = np.zeros(self.channels, dtype=float)
    self.initialize()
    self.decay = 0.8 
    self.mean_run = None 
    self.var_run = None 
    self.mean_batch = None
    self.var_batch = None
    self.optimizerval=None

  def initialize(self, weight_init=None, bais_init=None):
      self.weights = np.ones(self.channels)
      self.bias = np.zeros(self.channels)

  def forward(self, input_tensor):
    if input_tensor.ndim == 4:
      self.convflag = True
      self.input_tensor = self.reformat(input_tensor)
    else:
      self.convflag = False
      self.input_tensor = input_tensor
    
    if self.testing_phase :
      self.mean_batch = self.mean_run
      self.var_batch = self.var_run
    else:
      self.mean_batch = np.average(self.input_tensor, axis=0,keepdims=True)
      self.var_batch = np.var(self.input_tensor, axis=0,keepdims=True)
      
      if self.mean_run is None and self.var_run is None:
        self.mean_run = self.mean_batch
        self.var_run = self.var_batch
      else:
        self.mean_run = self.decay*self.mean_run + (1-self.decay)*self.mean_batch
        self.var_run = self.decay*self.var_run + (1-self.decay)*self.var_batch
              
    self.norm_input = (self.input_tensor - self.mean_batch) / np.sqrt(self.var_batch + np.finfo(float).eps)
      
    norm_out = self.weights * self.norm_input + self.bias
    
    if self.convflag :
      return self.reformat(norm_out)
      
    return norm_out

  def backward(self, error_tensor):
    if self.convflag :
      error_tensor = self.reformat(error_tensor)
    
    self.gradient_weights = np.sum(error_tensor * self.norm_input, axis=0)
    self.gradient_bias = np.sum(error_tensor, axis=0)
    
    if self.optimizerval:
      self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
      self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)
    
    error_tensor_prev = \
      compute_bn_gradients(error_tensor, self.input_tensor,\
                            self.weights, self.mean_batch, self.var_batch) 
    if self.convflag :
      error_tensor_prev = self.reformat(error_tensor_prev)
    
    return error_tensor_prev
    
  def reformat(self, input_tensor):
    if input_tensor.ndim == 4: 
      b, h, _, _ = self.input_tensor_shape = input_tensor.shape
      input_tensor = input_tensor.reshape(b, h, -1) # b x h x m.n
      input_tensor = np.swapaxes(input_tensor, 1, 2) # b x m.n x h
      return input_tensor.reshape(-1, h) # b.m.n x h
    
    elif input_tensor.ndim == 2 : #undo the reformat
      b, h, _, _ = self.input_tensor_shape
      input_tensor = input_tensor.reshape(b, -1, h) # b x m.n x h
      input_tensor = np.swapaxes(input_tensor, 1, 2) # b x h x m.n
      return input_tensor.reshape(*self.input_tensor_shape) # b x h x m x n
    
    else:
      return input_tensor
    
  @property
  def optimizer(self):
    return self.optimizerval
  @optimizer.setter
  def optimizer(self, newValue):
    self.optimizerval = newValue

  @property
  def gradient_weights(self):
    return self.grad_weightsval
  @gradient_weights.setter
  def gradient_weights(self, newValue):
    self.grad_weightsval = newValue

  @property
  def gradient_bias(self):
    return self.grad_biasval
  @gradient_bias.setter
  def gradient_bias(self, newValue):
    self.grad_biasval = newValue

