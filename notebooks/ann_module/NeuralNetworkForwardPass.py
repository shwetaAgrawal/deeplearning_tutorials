import numpy as np

class NeuralNetworkLayer:
  """
  A simple neural network layer implementation.
  """

  def __init__(self, input_dim: int, output_dim: int, act_function: callable, is_bias: bool = True) -> None:
    """
    Initialize the layer with input and output dimensions, activation function
    Weights will be initialized randomly for now it could be either -1, 0 or 1
    Bias will be initialized randomly for now binary values - 0 or 1

    Args:
      input_dim (int) : Number of input dimensions
      output_dim (int) : Number of output dimensions
      act_function (function) : Activation function to use
      is_bias (bool) : Whether to use bias or not
    """
    self.input_dimensions = input_dim
    self.output_dimensions = output_dim
    self.activation_function = act_function

    # For now we can initialize weights randomly to start with
    # we will deep dive later on how to set initial weights while covering the model training
    self.weights = np.random.randint(-1, 2, size=(self.input_dimensions, self.output_dimensions))
    if is_bias:
      self.bias = np.random.randint(0, 2, size=(self.output_dimensions, 1))
    else:
      self.bias = np.zeros((self.output_dimensions, 1))


  def forward(self, input:np.ndarray) -> np.ndarray:
    """
    Calculate the output of the layer for the given input.

    Args:
      input (np.ndarray): The input to the layer.

    Returns:
      np.ndarray: The output of the layer.
    """
    return self.activation_function(np.matmul(self.weights.T, input) + self.bias)
  
class NeuralNetworkForwardPass:
    def __init__(self, input_dimensions: int, output_dimensions: int, 
                 output_layer_actfn: callable,
                 layer_neuron_count_actfn: list[tuple[int, callable]]) -> None:
        """
        Initialize the neural network with input and output dimensions and number of neurons in each hidden layer
        Weights will be initialized randomly for now (binary values)
        Bias will be initialized randomly for now (-1, 0, 1)
        Lets assume that we are using only ReLU activation function for now for all layers

        Args:
            input_dimensions (int) : Number of input dimensions
            output_dimensions (int) : Number of output dimensions
            layer_neuron_count_actfn (list[tuple]) : Number of neurons in each hidden layer
        """
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.layer_neuron_count_actfn = layer_neuron_count_actfn
        self.layers = []
        
        tmp_input_dimensions = input_dimensions
        for num_neuron, activation_function in layer_neuron_count_actfn:
            self.add_layer(tmp_input_dimensions, num_neuron, activation_function)
            tmp_input_dimensions = num_neuron
        self.add_layer(tmp_input_dimensions, output_dimensions, output_layer_actfn)

    def add_layer(self, input_dimensions: int, output_dimensions: int, activation_function: callable) -> None:
        """
        Add a layer to the neural network.

        Args:
            input_dimensions (int) : Number of input dimensions
            output_dimensions (int) : Number of output dimensions
        """
        if input_dimensions <= 0 or output_dimensions <= 0:
            raise ValueError("Number of neurons in hidden layer should be greater than 0")
        self.layers.append(NeuralNetworkLayer(input_dimensions, output_dimensions, activation_function, is_bias=True))

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Calculate the output of the neural network for the given input.

        Args:
            input (np.ndarray): The input to the neural network.

        Returns:
            np.ndarray: The output of the neural network.
        """
        for layer in self.layers:
            input = layer.forward(input)
        return input