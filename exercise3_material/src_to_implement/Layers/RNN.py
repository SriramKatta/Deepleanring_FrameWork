from Layers.Base import BaseLayer

class RNN(BaseLayer):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.trainable = True
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.initialize()
    self.memval = False
    self.optimizerval = None

  
  def initialize(self, weights_initializer, bias_initializer):
    self.weights = weights_initializer.initialize()
    self.bias = bias_initializer.initialize()

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