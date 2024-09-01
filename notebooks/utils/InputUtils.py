import numpy as np

class InputUtils:
  def __init__(self, input_dimensions: int):
    self.input_dimensions = input_dimensions

  def getInput(self):
    return self.getInputBatch(1)

  def getInputBatch(self, batch_size: int):
    x = np.random.randint(0, 5, size=(self.input_dimensions, batch_size))
    return x
  
class SampleInputOutputUtils:
  def __init__(self, input_dimensions: int, output_dimensions: int):
    self.input_dimensions = input_dimensions
    self.output_dimensions = output_dimensions
  
  def getSampleInputOutput(self):
    return self.getSampleInputOutputBatch(1)
  
  def getSampleInputOutputBatch(self, batch_size: int):
    x = np.random.randint(0, 5, size=(self.input_dimensions, batch_size))
    y = np.random.randint(0, 2, size=(self.output_dimensions, batch_size))
    return x, y
