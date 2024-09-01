"""Implementing common activation functions. Expected input to all the functions are numpy arrays"""
import numpy as np

#TODO: Implement the gradient of the activation functions

class ActivationFunctions:
  @staticmethod
  def linear(x):
    """Linear activation function: f(x) = x
    Args:
      x: np.ndarray : input to the activation function
    
    Returns:
      np.ndarray
    """
    return x

  @staticmethod
  def sigmoid(x):
    """Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
    
    Args:
      x: np.ndarray : input to the activation function
    
    Returns:
      np.ndarray
    """
    return 1 / (1 + np.exp(-x))

  @staticmethod
  def relu(x):
    """ReLU activation function: f(x) = max(0, x)
    
    Args:
      x: np.ndarray : input to the activation function
    
    Returns:
      np.ndarray
    """
    # return max(0, x) this won't work for an array
    return np.maximum(0, x)

  @staticmethod
  def tanh(x):
    """Tanh activation function: f(x) = (exp(2x) - 1) / (exp(2x) + 1)
    
    Args:
      x: np.ndarray : input to the activation function
    
    Returns:
      np.ndarray
    """
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
  
  @staticmethod
  def softmax(x):
    """Softmax activation function: f(x) = exp(x) / sum(exp(x))

    Args:
        x: np.ndarray : input to the activation function

    Returns:
        np.ndarray
    """
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)