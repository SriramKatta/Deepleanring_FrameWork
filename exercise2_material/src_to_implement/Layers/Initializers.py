import numpy as np

class Constant:
  def __init__(self, val=0.1):
    self.const_val = val
  def initialize(self, weights_shape, fan_in, fan_out):
    return np.full(weights_shape,self.const_val)

class UniformRandom:
  def initialize(self, weights_shape, fan_in, fan_out):
    return np.random.uniform(0.0,1.0,weights_shape)

class Xavier:
  def initialize(self, weights_shape, fan_in, fan_out):
    sigma = np.sqrt(2/(fan_in + fan_out))
    return np.random.normal(0.0,sigma, weights_shape)

class He:
  def initialize(self, weights_shape, fan_in, fan_out):
    sigma = np.sqrt(2/(fan_in))
    return np.random.normal(0.0,sigma, weights_shape)
    