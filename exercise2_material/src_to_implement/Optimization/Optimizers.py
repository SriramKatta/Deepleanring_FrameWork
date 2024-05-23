import numpy as np

class Sgd:
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate

  def calculate_update(self, weight_tensor, gradient_tensor):
    return weight_tensor - self.learning_rate * gradient_tensor
    
class SgdWithMomentum:
  def __init__(self,learning_rate, momentum_rate):
    self.learning_rate = learning_rate
    self.momentum_rate = momentum_rate
    self.old_update_tensor = 0
    
  def calculate_update(self, weight_tensor, gradient_tensor):
    self.old_update_tensor = self.momentum_rate * self.old_update_tensor - self.learning_rate * gradient_tensor
    return weight_tensor + self.old_update_tensor

class Adam:
  def __init__(self, learning_rate, mu, rho):
    self.learning_rate = learning_rate
    self.mu = mu
    self.rho = rho
    self.gk = 0
    self.vk = 0
    self.rk = 0
    self.k = 1

  def calculate_update(self, weight_tensor, gradient_tensor):
    self.gk = gradient_tensor
    self.vk = self.mu * self.vk + (1 - self.mu) * self.gk
    self.rk = self.rho * self.rk + (1 - self.rho) * np.multiply(self.gk, self.gk)
    vkhat = self.vk / (1 - self.mu**self.k)
    rkhat = self.rk / (1 - self.rho**self.k)
    self.k += 1
    return weight_tensor - self.learning_rate * (vkhat / (np.sqrt(rkhat) + np.finfo(float).eps))