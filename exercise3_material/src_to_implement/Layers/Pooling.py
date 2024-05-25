from Layers.Base import BaseLayer

import numpy as np

class Pooling(BaseLayer):
  def __init__(self, stride_shape, pooling_shape):
    super().__init__()
    self.stride_shape = stride_shape
    self.pooling_shape = pooling_shape
    self.max_pos = []
  
  def forward(self, input_tensor):
    _, _, rows, cols = self.input_tensor_shape = input_tensor.shape
    poolrows, poolcols = self.pooling_shape
    strirows, stricols = self.stride_shape
    outrows = ((rows - poolrows) // strirows) + 1
    outcols = ((cols - poolcols) // stricols) + 1
    output_tensor = np.zeros((*self.input_tensor_shape[:2], outrows, outcols))
    for batch_num, cyxmat in enumerate(output_tensor):
      for channel_num, out_yxmat in enumerate(cyxmat):
        in_row = 0
        for out_row in range(outrows):
          in_col = 0
          for out_col in  range(outcols):
            inmat = input_tensor[batch_num, channel_num, in_row : in_row + poolrows, in_col : in_col + poolcols]
            out_yxmat[out_row, out_col] = np.max(inmat)
            maxlocrow, maxloccol = np.unravel_index(np.argmax(inmat), inmat.shape)
            self.max_pos.append([batch_num, channel_num, in_row + maxlocrow, in_col + maxloccol])
            in_col += stricols
          in_row += strirows
    return output_tensor
  
  def backward(self, error_tensor):
    error_output = np.zeros(self.input_tensor_shape)
    for i in range(error_tensor.size):
      error_output[*self.max_pos[i]] += error_tensor[np.unravel_index(i, error_tensor.shape)]
    return error_output
