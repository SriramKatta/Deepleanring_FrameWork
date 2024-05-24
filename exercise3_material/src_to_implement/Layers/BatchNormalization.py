from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients
import numpy as np

class BatchNormalization(BaseLayer):
  def __init__(self, channels):
    super().__init__()
    self.trainable = True
    self.channels = channels
    self.initialize()
    #try setiing to zero to remove a if else
    self.mean_batch = None
    self.var_batch = None
    self.mean_run = None
    self.var_run = None
    self.decay = 0.8
    self.grad_weightsval = None
    self.grad_biasval = None
    self.convflag = False
    self.weightoptimizerval = None
    self.biasoptimizerval = None

  def initialize(self):
    self.weights = np.ones(self.channels, dtype=float)
    self.bias = np.zeros(self.channels, dtype=float)

  def forward(self, input_tensor):
    if input_tensor.ndim == 4:
      self.input_tensor = self.reformat(input_tensor)
      self.convflag = True
    else:
      self.input_tensor = input_tensor
      self.convflag = False
    
    if self.testing_phase:
      self.mean_batch = self.mean_run
      self.var_batch = self.var_run
    else:
      self.mean_batch = self.input_tensor.mean(axis=0,keepdims=True)
      self.var_batch = self.input_tensor.var(axis=0, keepdims=True)
      if self.mean_run is None and self.var_run is None:
        self.mean_run = self.mean_batch
        self.var_run = self.var_batch
      else :
        self.mean_run = self.decay * self.mean_run + (1-self.decay) * self.mean_batch
        self.var_batch = self.decay * self.var_run + (1-self.decay) * self.var_batch

    self.norm_input = (input_tensor - self.mean_batch) / np.sqrt(self.var_batch + np.finfo(float).eps)
    norm_output = self.weights * self.norm_input + self.bias

    if self.convflag:
      return self.reformat(norm_output)

    return norm_output

  def backward(self, error_tensor):
    self.grad_weightsval = np.sum(error_tensor * self.norm_input, axis=0)
    self.grad_biasval = np.sum(error_tensor, axis=0)
    if self.weightoptimizerval is not None:
      self.weights = self.weightoptimizerval.calculate_update(self.weights, self.grad_weightsval)
    if self.biasoptimizerval is not None:
      self.bias = self.biasoptimizer.calculate_update(self.bias, self.grad_biasval)
    if error_tensor.ndim == 4:
      error_tensor = self.reformat(error_tensor)  
    error_tensor_prev = compute_bn_gradients(error_tensor,self.norm_input, self.weights, self.mean_batch, self.var_batch)
    if self.convflag :
      return self.reformat(error_tensor_prev)
    return error_tensor_prev


  def reformat(self, input_tensor):
    if input_tensor.ndim == 4: 
      b, h, _, _ = self.input_tensor_shape = input_tensor.shape
      input_tensor = input_tensor.reshape(b, h, -1) # b x h x m.n
      input_tensor = np.swapaxes(input_tensor, 1, 2) # b x m.n x h
      return input_tensor.reshape(-1, h) # b.m.n x h
    
    elif input_tensor.ndim == 2 : #undo the reformat
      b, h, m, n = self.input_tensor_shape
      input_tensor = input_tensor.reshape(b, -1, h) # b x m.n x h
      input_tensor = np.swapaxes(input_tensor, 1, 2) # b x h x m.n
      return input_tensor.reshape(*self.input_tensor_shape) # b x h x m x n
    
    else:
      return input_tensor
    
  @property
  def optimizer(self):
    return self.optimizerval
  
  @optimizer.setter
  def optimizer(self,newval):
    self.optimizerval = newval

  @property
  def bias_optimizer(self):
    return self.biasoptimizer
  
  @bias_optimizer.setter
  def bias_optimizer(self,newval):
    self.biasoptimizer = newval

  @property
  def gradient_weights(self):
    return self.grad_weightsval
  
  @gradient_weights.setter
  def gradient_weights(self,newval):
    self.grad_weightsval = newval

  @property
  def gradient_bias(self):
    return self.grad_biasval
  
  @gradient_bias.setter
  def gradient_bias(self,newval):
    self.grad_biasval = newval






