import pattern as pt
import generator 

def mainCheckers():
  vis = pt.Checker(250, 25)
  vis.draw()
  vis.show()

def mainCircle():
  vis = pt.Circle(1024, 200, (512, 256))
  vis.draw()
  vis.show()

def mainSpectrum():
  vis = pt.Spectrum(250)
  vis.draw()
  vis.show()

def maingenerator():
  vis = generator.ImageGenerator('./exercise_data/', './Labels.json', 10, [32, 32, 3], rotation=False, mirroring=False,
                             shuffle=True)
  vis.show()
  


if __name__ == '__main__':
  mainCheckers()
  #mainCircle()
  #mainSpectrum()
  #maingenerator()