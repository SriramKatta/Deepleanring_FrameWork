import numpy as np

class Optimizer:
  def __init__(self):
    self.regularizer = None

  def add_regularizer(self,regularizer):
    self.regularizer = regularizer

class Sgd(Optimizer):
  def __init__(self, learning_rate):
    super().__init__()
    self.learning_rate = learning_rate

  def calculate_update(self, weight_tensor, gradient_tensor):
    if self.regularizer is None:
      regterm = 0
    else:
      regterm = self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
    return weight_tensor - self.learning_rate * gradient_tensor - regterm
    
class SgdWithMomentum(Optimizer):
  def __init__(self,learning_rate, momentum_rate):
    super().__init__()
    self.learning_rate = learning_rate
    self.momentum_rate = momentum_rate
    self.old_update_tensor = 0
    
  def calculate_update(self, weight_tensor, gradient_tensor):
    if self.regularizer is None :
      regterm = 0
    else :
      regterm = self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
    self.old_update_tensor = self.momentum_rate * self.old_update_tensor - self.learning_rate * gradient_tensor
    return weight_tensor + self.old_update_tensor - regterm

class Adam(Optimizer):
  def __init__(self, learning_rate, mu, rho):
    super().__init__()
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
    if self.regularizer is None :
      regterm = 0
    else :
      regterm = self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
    return weight_tensor - self.learning_rate * (vkhat / (np.sqrt(rkhat) + np.finfo(float).eps)) - regterm