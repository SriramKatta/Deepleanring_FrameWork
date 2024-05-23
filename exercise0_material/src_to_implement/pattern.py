import numpy as np
import matplotlib.pyplot as plt


class Checker:
  def __init__(self, res, ts):
    self.res = res
    self.ts = ts

  def draw(self):
    if(self.res % (2*self.ts) != 0):
      exit()
    block = np.ones((2*self.ts,2*self.ts),dtype=int)
    block[:self.ts,:self.ts] = 0
    block[self.ts:,self.ts:] = 0
    self.output = np.tile(block,(self.res // (2*self.ts), self.res //(2*self.ts)))
    return self.output.copy()

  def show(self):
    plt.imshow(self.output,cmap='gray')
    plt.show()

class Circle:
  def __init__(self, res, radius, position):
    self.res = res
    self.radius = radius
    self.position = position

    self.output = np.zeros((self.res,self.res),dtype=int)
  
  def draw(self):
    sampleloc = np.linspace(0, self.res, self.res + 1, dtype=int)
    row, col = np.meshgrid(sampleloc,sampleloc)

    distbool = np.sqrt((row - self.position[1])**2 + (col - self.position[0])**2) <= self.radius
    self.output[row[distbool], col[distbool]] = 1

    return self.output.copy()

  def show(self):
    plt.imshow(self.output,cmap='gray')
    plt.show() 

class Spectrum:
  def __init__(self, res):
    self.res = res
    self.output = np.zeros((res,res,3),dtype=float)

  def draw(self):
    forwpatt = np.linspace(0.0,1.0, self.res, dtype=float)
    backpatt = np.linspace(1.0,0.0, self.res, dtype=float)
    self.output[:,:,0] = np.tile(forwpatt, (self.res, 1))
    self.output[:,:,1] = np.tile(forwpatt, (self.res, 1)).T
    self.output[:,:,2] = np.tile(backpatt, (self.res, 1))

    return self.output.copy()
  
  def show(self):
    plt.imshow(self.output)
    plt.show() 
    