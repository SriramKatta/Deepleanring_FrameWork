from Layers.Base import BaseLayer
import numpy as np
from scipy.signal import convolve,correlate
import copy

class Conv(BaseLayer):
  def __init__(self, stride_shape, convolution_shape, num_kernels):
    super().__init__()
    self.trainable = True
    self.stride_shape = stride_shape
    self.convolution_shape = convolution_shape # 1D : [c,m], 2D : [c,m,n]
    self.num_kernels = num_kernels
    self.stride_2dim = bool(len(self.stride_shape) == 2)
    self.kernels = np.random.rand(self.num_kernels, *self.convolution_shape)
    self.bias = np.random.rand(self.num_kernels)
    self.gradient_weights_val = None
    self.gradient_bias_val = None
    self.weightoptimizerval = None
    self.biasoptimizerval = None

  
  def forward(self, input_tensor):
    self.input_tensor = input_tensor #1D : [b c y] ; 2D:[b c y x]
    self.batchsize, self.channels, *spatial_dimensions  = self.input_tensor.shape
  
    self.output_tensor = np.zeros((self.batchsize, self.num_kernels, *spatial_dimensions))

    for batch_num, image in enumerate(self.input_tensor):
       for kernel_num, kernel in enumerate(self.kernels):
          self.output_tensor[batch_num, kernel_num] = correlate(image, kernel, mode='same')[self.channels//2]
          self.output_tensor[batch_num, kernel_num] += self.bias[kernel_num]
    
    if(self.stride_2dim):
      return self.output_tensor[:,:,::self.stride_shape[0],::self.stride_shape[1]]
    else:
      return self.output_tensor[:,:,::self.stride_shape[0]]

  def backward(self, error_tensor):
    upsampled_error = np.zeros_like(self.output_tensor)
    if self.stride_2dim:
      upsampled_error[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor
    else:
      upsampled_error[:, :, ::self.stride_shape[0]] = error_tensor

    gradient_kernel = np.swapaxes(self.kernels,1,0)
    gradient_kernel = np.fliplr(gradient_kernel)

    error_tensor_prev = np.zeros_like(self.input_tensor)

    for ele_num, ele in enumerate(upsampled_error):
       for kernel_num, kernel in enumerate(gradient_kernel):
          error_tensor_prev[ele_num,kernel_num] = convolve(ele,kernel, mode="same")[self.num_kernels//2] 
    
    if self.stride_2dim:
      pad_left = (self.convolution_shape[1]-1)//2
      pad_right = self.convolution_shape[1]//2
      pad_top = (self.convolution_shape[2]-1)//2
      pad_bottom = self.convolution_shape[2]//2
      self.input_tensor = np.pad(self.input_tensor, ((0,0),(0,0),(pad_left, pad_right),(pad_top, pad_bottom)))
    else :
      pad_left = (self.convolution_shape[1]-1)//2
      pad_right = (self.convolution_shape[1])//2
      self.input_tensor = (np.pad(self.input_tensor, ((0,0),(0,0),(pad_left, pad_right))))

    self.gradient_weights_val = np.zeros_like(self.kernels)
    self.gradient_bias_val = np.zeros_like(self.bias)

    for ele_num, error_ele in enumerate(upsampled_error):
      for error_channel_num, error_channel in enumerate(error_ele):
        for input_channel_ctr in range(self.convolution_shape[0]):
          self.gradient_weights_val[error_channel_num,input_channel_ctr] += \
                    correlate(self.input_tensor[ele_num,input_channel_ctr], error_channel, mode='valid')
        self.gradient_bias_val[error_channel_num] += np.sum(error_channel)

    if self.weightoptimizerval is not None :
        self.kernels = self.weightoptimizerval.calculate_update(self.kernels, self.gradient_weights_val)
    if self.biasoptimizerval is not None:
        self.bias = self.biasoptimizerval.calculate_update(self.bias, self.gradient_bias_val)
    
    return error_tensor_prev

  def initialize(self, weights_initializer, bias_initializer):
    fan_in = np.prod(self.convolution_shape)
    fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
    self.kernels = weights_initializer.initialize(self.kernels.shape, fan_in, fan_out)
    self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

  @property
  def gradient_weights(self):
    return self.gradient_weights_val
  
  @gradient_weights.setter
  def gradient_weights(self, gradweightnewValue):
    self.gradient_weights_val = gradweightnewValue
  
  @property
  def gradient_bias(self):
    return self.gradient_bias_val
  
  @gradient_bias.setter
  def gradient_bias(self, gradbiasnewValue):
    self.gradient_bias_val = gradbiasnewValue
  
  @property
  def optimizer(self):
    return self.weightoptimizerval
  
  @optimizer.setter
  def optimizer(self, optimizerval):
    self.weightoptimizerval = copy.deepcopy(optimizerval)
    self.biasoptimizerval = copy.deepcopy(optimizerval)

  @property
  def weights(self):
    return self.kernels
  
  @weights.setter
  def weights(self, newval):
    self.kernels = newval


