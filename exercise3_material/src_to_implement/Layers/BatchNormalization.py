from Layers.Base import BaseLayer
import numpy as np

class BatchNormalization(BaseLayer):
  def __init__(self, channels):
    super().__init__()
    self.trainable = True
    self.channels = channels
    self.initialize()
    
  def initialize(self):
    self.weights = np.ones(self.channels)
    self.bias = np.zeros(self.channels)

  def forward(self, input_tensor):
    pass

  def backward(self, error_tensor):
    pass

