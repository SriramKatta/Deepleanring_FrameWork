from Layers.Base import BaseLayer
import numpy as np

class Dropout(BaseLayer):
  def __init__(self, probability):
    super().__init__()
    self.probability = probability

  def forward(self, input_tensor):
    if self.testing_phase:
      return input_tensor
    self.mask = np.random.rand(*input_tensor.shape) < self.probability
    return input_tensor * self.mask / self.probability

  def backward(self, error_tensor):
    if self.testing_phase:
      return error_tensor
    return error_tensor * self.mask / self.probability