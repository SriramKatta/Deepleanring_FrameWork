from Layers.Base import BaseLayer

import numpy as np

class SoftMax(BaseLayer):
  def __init__(self):
    super().__init__()
    self.result_tensor = None

  def forward(self, input_tensor):
    self.result_tensor = np.exp(input_tensor - np.max(input_tensor)) 
    self.result_tensor = self.result_tensor / np.sum(self.result_tensor, axis=1, keepdims=True)
    return self.result_tensor
  
  def backward(self,error_tensor):
    return self.result_tensor * (error_tensor - np.sum(error_tensor*self.result_tensor, axis=1,keepdims=1))
